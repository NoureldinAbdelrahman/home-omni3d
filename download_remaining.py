#!/usr/bin/env python3
"""Resumable, verbose downloader for the REMAINING OmniObject3D categories.

Fetches only the two modalities we use, for every category on the remote that is
not already one of our 76 house/bedroom categories:

* ``blender_renders_24_views``  - 24 renders/object (one tar.gz per category)
* ``point_clouds`` 4096         - named per-object PLYs (one archive, all categories)

Differences from ``download_house_extra.py`` (which stays as the simple path):

* **Resumable at every level.** Archives are fetched in fixed-size chunks into
  ``<name>.part``; completed chunks are journaled in ``<name>.part.json``. After
  a crash, Ctrl+C, reboot or network drop, re-running the same command re-fetches
  only the chunks that were missing (at most a few MiB per worker is lost). Finished
  categories are never touched again; an archive already on disk is re-extracted
  rather than re-downloaded.
* **Verified.** Every archive is checked against the remote sha256 before it is
  extracted; extraction is staged and swapped in with an atomic rename, so
  ``renders/<cat>/`` is either complete or absent, never half-written.
* **Verbose.** One live status line: overall percent / bytes / current speed /
  session-average speed / ETA / elapsed, plus the current file's progress.
  Per-category completion lines are printed permanently and mirrored to a log file.

Output layout (same as the other downloaders)::

    <output-dir>/renders/<cat>/<cat>_NNN/{000..023.png, transforms.json}
    <output-dir>/point_clouds/<cat>/<cat>_NNN.npy        (float32, 4096 x 3)
    <output-dir>/.download_state.json                    (what is finished)
    <output-dir>/_downloads/  _extract/                  (scratch, auto-cleaned)

Usage::

    python download_remaining.py --output-dir dataset --dry-run   # plan + sizes only
    python download_remaining.py --output-dir dataset             # the real thing
    python download_remaining.py --output-dir dataset --categories laptop   # repair one
    python download_remaining.py --output-dir dataset --modality renders --workers 12

Re-run the exact same command to resume. Ctrl+C once = graceful stop (state saved);
twice = hard exit (state is still safe, you just lose up to a few seconds).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import errno
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import signal
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import requests

import download_bedroom_subset as base
from bedroom_categories import resolve_category
from house_extra_categories import ALL_HOUSE_CATEGORIES

GiB = 1024**3
MiB = 1024**2
VIEWS_PER_OBJECT = 24
STATE_NAME = ".download_state.json"
URL_TTL = 600  # signed URLs are refreshed after 10 min or on a 401/403
BACKOFF_BASE = 2.0  # retry delay: 2s, 4s, 8s ... capped at BACKOFF_CAP (+-25% jitter)
BACKOFF_CAP = 60.0


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TiB"


def fmt_rate(bps: float | None) -> str:
    return "   n/a   " if bps is None else f"{fmt_bytes(bps)}/s"


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec != sec or sec == float("inf") or sec < 0:
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


class Stopped(Exception):
    """Graceful stop requested (Ctrl+C); state has been / will be saved."""


class Fatal(Exception):
    """Not worth retrying (disk full, server ignores Range, ...)."""


class VerifyError(Exception):
    """sha256 mismatch on a fully downloaded archive."""


class TransientError(Exception):
    """Retryable remote problem (URL resolution failed, bad status, short read)."""


# --------------------------------------------------------------------------- #
# console + log: one live status line, permanent lines scroll above it
# --------------------------------------------------------------------------- #
class Console:
    def __init__(self, log_path: Path | None):
        self.lock = threading.Lock()
        # keep our own handle: UrlProvider temporarily redirects sys.stdout to swallow SDK noise
        self.out = sys.stdout
        self.tty = self.out.isatty()
        self.log_file = open(log_path, "a", buffering=1) if log_path else None

    def log(self, msg: str, *, screen: bool = True) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        with self.lock:
            if self.log_file:
                self.log_file.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")
            if screen:
                if self.tty:
                    self.out.write("\r\033[K")
                self.out.write(f"{stamp} {msg}\n")
                self.out.flush()

    def status(self, line: str) -> None:
        with self.lock:
            if self.tty:
                width = shutil.get_terminal_size((120, 20)).columns
                self.out.write("\r\033[K" + line[: width - 1])
            else:
                self.out.write(line + "\n")
            self.out.flush()

    def clear_status(self) -> None:
        with self.lock:
            if self.tty:
                self.out.write("\r\033[K")
                self.out.flush()


# --------------------------------------------------------------------------- #
# progress accounting + reporter thread
# --------------------------------------------------------------------------- #
class Progress:
    """Thread-safe byte counters behind the status line.

    ``total`` covers the whole job (finished categories count as done at start),
    so the overall percentage keeps meaning across restarts. Speeds and ETA only
    use bytes moved in this session.
    """

    WINDOW = 10.0  # seconds for the "current speed"

    def __init__(self, total: int, initial_done: int, n_items: int, items_done: int):
        self.lock = threading.Lock()
        self.total = total
        self.done = initial_done
        self.initial = initial_done
        self.resumed = 0  # bytes found already on disk in .part files this session
        self.t0 = time.monotonic()
        self.n_items, self.items_done = n_items, items_done
        self.label = ""
        self.phase = "starting"
        self.phase_pct: float | None = None
        self.file_total = 0
        self.file_done = 0
        self.file_resumed = 0
        self.samples: collections.deque[tuple[float, int]] = collections.deque()

    # -- called by workers
    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            self.file_done += n

    # -- called by the main flow
    def begin_file(self, label: str, size: int, already: int, phase: str = "downloading") -> None:
        with self.lock:
            self.label, self.file_total, self.file_done = label, size, already
            self.file_resumed = already
            self.done += already
            self.resumed += already
            self.phase, self.phase_pct = phase, None

    def end_file(self, success: bool, *, file_bytes_in_total: int) -> None:
        """Success: the file is complete. Failure: roll its bytes back out of ``done``."""
        with self.lock:
            if success:
                self.done += file_bytes_in_total - self.file_done
            else:
                self.done -= self.file_done
                self.resumed -= self.file_resumed
            self.file_done = 0
            self.file_total = 0
            self.file_resumed = 0

    def set_phase(self, phase: str, pct: float | None = None) -> None:
        with self.lock:
            self.phase, self.phase_pct = phase, pct

    def item_finished(self) -> None:
        with self.lock:
            self.items_done += 1

    # -- called by the reporter
    def sample(self) -> float | None:
        now = time.monotonic()
        with self.lock:
            self.samples.append((now, self.done))
            while len(self.samples) > 2 and now - self.samples[0][0] > self.WINDOW:
                self.samples.popleft()
            (t_a, d_a), (t_b, d_b) = self.samples[0], self.samples[-1]
            return (d_b - d_a) / (t_b - t_a) if t_b - t_a > 0.5 else None

    def render(self, rate_now: float | None) -> str:
        with self.lock:
            elapsed = time.monotonic() - self.t0
            sess = self.done - self.initial - self.resumed
            avg = sess / elapsed if elapsed > 3 and sess > 0 else None
            remaining = max(0, self.total - self.done)
            eta = remaining / avg if avg else None
            pct = 100.0 * self.done / self.total if self.total else 100.0
            filled = int(round(pct / 100 * 20))
            bar = "█" * filled + "░" * (20 - filled)
            head = (
                f"ALL {pct:5.1f}% [{bar}] {fmt_bytes(self.done)}/{fmt_bytes(self.total)}"
                f" | now {fmt_rate(rate_now)} avg {fmt_rate(avg)}"
                f" | ETA {fmt_dur(eta)} (elapsed {fmt_dur(elapsed)})"
            )
            idx = f"[{min(self.items_done + 1, self.n_items)}/{self.n_items}]"
            if self.phase == "downloading" and self.file_total:
                fpct = 100.0 * self.file_done / self.file_total
                feta = (self.file_total - self.file_done) / rate_now if rate_now else None
                tail = (
                    f" | {idx} {self.label} {fpct:5.1f}% "
                    f"{fmt_bytes(self.file_done)}/{fmt_bytes(self.file_total)} ETA {fmt_dur(feta)}"
                )
            else:
                p = f" {self.phase_pct:.0f}%" if self.phase_pct is not None else ""
                tail = f" | {idx} {self.label} {self.phase}{p}"
            return head + tail


class Reporter(threading.Thread):
    def __init__(self, prog: Progress, console: Console, interval: float):
        super().__init__(daemon=True)
        self.prog, self.console, self.interval = prog, console, interval
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.console.status(self.prog.render(self.prog.sample()))

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=2)
        self.console.clear_status()


# --------------------------------------------------------------------------- #
# remote listing + signed URLs (same OpenXLab API the SDK uses)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Remote:
    name: str  # path without leading slash, as the resolve endpoint wants it
    size: int
    sha256: str
    dataset_id: int


def list_remote(prefix: str) -> list[Remote]:
    api = base._client().get_api()
    dataset_name = base.DATASET_REPO.replace("/", ",")
    out: list[Remote] = []
    after = None
    try:
        while True:
            page = api.get_dataset_files(
                dataset_name=dataset_name,
                payload={"prefix": prefix},
                needContent=True,
                after=after,
                limit=500,
            )
            for it in page.get("list", []):
                path = it["path"]
                if path.startswith("//"):
                    path = path[1:]
                out.append(
                    Remote(
                        name=path.lstrip("/"),
                        size=int(it.get("size", 0)),
                        sha256=(it.get("sha256") or "").lower(),
                        dataset_id=int(it["dataset_id"]),
                    )
                )
            if not page.get("hasNext", False) or not page.get("after"):
                break
            after = page["after"]
    except SystemExit as exc:  # the SDK calls sys.exit on API errors
        raise RuntimeError(f"remote listing failed for {prefix!r} (SDK exit {exc})") from exc
    if not out:
        raise RuntimeError(f"no remote files under {prefix!r}")
    return out


class UrlProvider:
    """Hands out the signed download URL for one remote file; refreshes on expiry."""

    def __init__(self, remote: Remote, log: Callable[[str], None]):
        self.remote, self.log = remote, log
        self._lock = threading.Lock()
        self._url: str | None = None
        self._t = 0.0
        self._checked = False

    def get(self, stale: str | None = None) -> str:
        with self._lock:
            now = time.monotonic()
            if self._url is None or stale == self._url or now - self._t > URL_TTL:
                self._url = self._resolve()
                self._t = now
            return self._url

    def _resolve(self) -> str:
        # The SDK prints the server's error body (a whole HTML page for firewall blocks) and then
        # calls sys.exit(). Capture that output so we can classify the failure and keep it short.
        sdk_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(sdk_out):
                api = base._client().get_api()
                if not self._checked:
                    api.download_check(dataset_id=self.remote.dataset_id, path=self.remote.name)
                    self._checked = True
                return api.get_dataset_download_urls(
                    self.remote.dataset_id, {"name": self.remote.name}
                )
        except SystemExit as exc:
            raise classify_sdk_failure(sdk_out.getvalue(), exc, self.remote.name) from exc


def classify_sdk_failure(sdk_output: str, exc: BaseException, name: str) -> Exception:
    """Map the SDK's printed error to Fatal (retrying cannot help) or TransientError."""
    m = re.match(r"\s*(\d{3})\s*:", sdk_output)
    status = int(m.group(1)) if m else None
    blocked = "blocked as it may cause" in sdk_output  # Alibaba Cloud WAF block page
    if status == 405 or blocked:
        return Fatal(
            f"OpenDataLab's web firewall blocks the download-URL request for {name} "
            f"(HTTP {status or 405}). This is deterministic - retrying will not help."
        )
    detail = re.sub(r"<[^>]+>|\s+", " ", sdk_output).strip()[:160]
    return TransientError(
        f"could not resolve download URL (HTTP {status or '?'}, SDK exit {exc})"
        + (f": {detail}" if detail else "")
    )


# --------------------------------------------------------------------------- #
# chunked, journaled, resumable file download
# --------------------------------------------------------------------------- #
_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = requests.Session()
    return s


def sha256_file(path: Path, prog: Progress | None = None) -> str:
    h = hashlib.sha256()
    size = max(1, path.stat().st_size)
    read = 0
    with open(path, "rb") as f:
        while block := f.read(8 * MiB):
            h.update(block)
            read += len(block)
            if prog:
                prog.set_phase("verifying sha256", 100.0 * read / size)
    return h.hexdigest()


def _verify(path: Path, remote: Remote, prog: Progress, console: Console) -> bool:
    if path.stat().st_size != remote.size:
        return False
    if not remote.sha256:
        console.log(f"  (no remote sha256 for {path.name}; size check only)")
        return True
    return sha256_file(path, prog) == remote.sha256


def download_file(
    remote: Remote,
    dest: Path,
    urls: UrlProvider,
    prog: Progress,
    console: Console,
    stop: threading.Event,
    *,
    label: str,
    workers: int,
    chunk: int,
    max_retries: int,
) -> None:
    """Download ``remote`` to ``dest`` (resuming), verify sha256, rename into place."""
    size = remote.size
    n_chunks = -(-size // chunk)
    part = dest.with_name(dest.name + ".part")
    meta = dest.with_name(dest.name + ".part.json")

    def chunk_len(i: int) -> int:
        return min(size, (i + 1) * chunk) - i * chunk

    # a fully downloaded archive left over from an interrupted extraction
    if dest.exists():
        prog.begin_file(label, size, size, phase="verifying sha256")
        if _verify(dest, remote, prog, console):
            console.log(f"  {dest.name}: complete archive already on disk, verified")
            return
        console.log(f"  {dest.name}: on-disk archive failed verification, re-downloading")
        prog.end_file(False, file_bytes_in_total=size)
        dest.unlink()

    done: set[int] = set()
    if part.exists() and meta.exists():
        try:
            m = json.loads(meta.read_text())
            if (
                m.get("size") == size
                and m.get("sha256") == remote.sha256
                and m.get("chunk") == chunk
                and part.stat().st_size == size
            ):
                done = {i for i in m["done"] if 0 <= i < n_chunks}
        except (ValueError, KeyError, OSError):
            done = set()
    if not done:
        with open(part, "wb") as f:  # sparse pre-allocation
            f.truncate(size)
        meta.unlink(missing_ok=True)
    already = sum(chunk_len(i) for i in done)
    if done:
        console.log(
            f"  resuming {label}: {len(done)}/{n_chunks} chunks "
            f"({fmt_bytes(already)}) already on disk"
        )
    prog.begin_file(label, size, already)

    fd = os.open(part, os.O_RDWR)
    abort = threading.Event()
    lock = threading.Lock()

    def should_stop() -> bool:
        return stop.is_set() or abort.is_set()

    def save() -> None:
        with lock:
            os.fsync(fd)  # data first, then the journal that claims it
            atomic_write_json(
                meta,
                {"size": size, "sha256": remote.sha256, "chunk": chunk, "done": sorted(done)},
            )

    def fetch(i: int) -> int:
        start = i * chunk
        end = start + chunk_len(i) - 1
        attempt = 0
        while True:
            if should_stop():
                raise Stopped()
            counted = 0
            url = None
            try:
                url = urls.get()
                r = _session().get(
                    url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(30, 60),
                )
                try:
                    if r.status_code in (400, 401, 403):
                        urls.get(stale=url)  # signed URL probably expired
                        raise TransientError(f"HTTP {r.status_code} (URL refreshed)")
                    if r.status_code == 200 and not (start == 0 and end == size - 1):
                        raise Fatal("server ignored the Range header (got 200 for a partial request)")
                    if r.status_code not in (200, 206):
                        raise TransientError(f"HTTP {r.status_code}")
                    pos = start
                    for data in r.iter_content(256 * 1024):
                        if should_stop():
                            raise Stopped()
                        os.pwrite(fd, data, pos)
                        pos += len(data)
                        counted += len(data)
                        prog.add(len(data))
                    if pos != end + 1:
                        raise TransientError(f"short read: {pos - start}/{end + 1 - start} bytes")
                finally:
                    r.close()
                return i
            except (Stopped, Fatal):
                prog.add(-counted)
                raise
            except OSError as exc:
                prog.add(-counted)
                if exc.errno == errno.ENOSPC:
                    raise Fatal("disk full") from exc
                err = exc
            except (requests.RequestException, TransientError, ValueError) as exc:
                prog.add(-counted)
                err = exc
            attempt += 1
            if max_retries and attempt > max_retries:
                raise Fatal(f"chunk {i} failed {attempt - 1}x, last error: {err}") from err
            delay = min(BACKOFF_CAP, BACKOFF_BASE * 2.0 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
            console.log(
                f"  ! {label} chunk {i}: {type(err).__name__}: {err} "
                f"- retry {attempt}/{max_retries or 'inf'} in {delay:.0f}s"
            )
            if stop.wait(delay) or abort.is_set():
                raise Stopped()

    todo = [i for i in range(n_chunks) if i not in done]
    futs: dict = {}
    last_save = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch, i): i for i in todo}
            try:
                for fut in as_completed(futs):
                    fut.result()
                    done.add(futs[fut])
                    if time.monotonic() - last_save > 3:
                        save()
                        last_save = time.monotonic()
            except BaseException:
                abort.set()
                for f in futs:
                    f.cancel()
                raise
    except BaseException:
        for f, i in futs.items():  # chunks that finished while we were unwinding
            if f.done() and not f.cancelled() and f.exception() is None:
                done.add(i)
        try:
            save()
        finally:
            os.close(fd)
        prog.end_file(False, file_bytes_in_total=size)
        raise
    save()
    os.close(fd)

    prog.set_phase("verifying sha256", 0)
    if not _verify(part, remote, prog, console):
        part.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        prog.end_file(False, file_bytes_in_total=size)
        raise VerifyError(f"{dest.name}: sha256/size mismatch after download")
    os.replace(part, dest)
    meta.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):  # py>=3.12 and backports
        tar.extract(member, dest, filter="data")
        return
    target = (dest / member.name).resolve()
    if not str(target).startswith(str(dest.resolve()) + os.sep) or member.issym() or member.islnk():
        raise Fatal(f"unsafe tar member: {member.name}")
    tar.extract(member, dest)


def extract_renders(
    archive: Path, cat: str, renders_root: Path, scratch: Path, prog: Progress
) -> tuple[int, int, list[str]]:
    """Extract a category archive and atomically swap it into ``renders/<cat>``."""
    work = scratch / cat
    shutil.rmtree(work, ignore_errors=True)
    raw, stage = work / "raw", work / "stage"
    raw.mkdir(parents=True)
    stage.mkdir()
    asize = max(1, archive.stat().st_size)
    with open(archive, "rb") as fh, tarfile.open(fileobj=fh, mode="r:gz") as tar:
        for k, member in enumerate(tar):
            _extract_member(tar, member, raw)
            if k % 50 == 0:
                prog.set_phase("extracting", 100.0 * fh.tell() / asize)
    # object dirs = directories holding 000.png (robust to an extra top-level dir)
    objects = sorted({p.parent for p in raw.rglob("000.png")})
    if not objects:
        raise Fatal(f"{archive.name}: no object folders (000.png) found")
    names = [o.name for o in objects]
    if len(set(names)) != len(names):
        raise Fatal(f"{archive.name}: duplicate object folder names")
    warnings: list[str] = []
    images = 0
    for o in objects:
        n_png = sum(1 for p in o.iterdir() if p.suffix == ".png")
        images += n_png
        if n_png != VIEWS_PER_OBJECT:
            warnings.append(f"{o.name}: {n_png} PNGs (expected {VIEWS_PER_OBJECT})")
        if not (o / "transforms.json").is_file():
            warnings.append(f"{o.name}: no transforms.json")
        os.replace(o, stage / o.name)
    target = renders_root / cat
    renders_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.replace(target, work / "old")
    os.replace(stage, target)
    shutil.rmtree(work, ignore_errors=True)
    return len(objects), images, warnings


def extract_point_clouds(
    archive: Path, wanted: set[str], pc_root: Path, prog: Progress
) -> dict[str, list[str]]:
    """Write ``<N>/<cat>/<obj>/pcd_<N>.ply`` as ``point_clouds/<cat>/<obj>.npy`` (float32)."""
    found: dict[str, list[str]] = {c: [] for c in wanted}
    asize = max(1, archive.stat().st_size)
    with open(archive, "rb") as fh, tarfile.open(fileobj=fh, mode="r:gz") as tar:
        for k, member in enumerate(tar):
            if k % 25 == 0:
                prog.set_phase("extracting clouds", 100.0 * fh.tell() / asize)
            parts = member.name.split("/")
            if not member.isfile() or len(parts) != 4 or not parts[3].endswith(".ply"):
                continue
            cat, obj = parts[1], parts[2]
            if cat not in wanted:
                continue
            cloud = base._read_ascii_ply(tar.extractfile(member).read())
            if cloud.ndim != 2 or cloud.shape[1] != 3:
                raise Fatal(f"unexpected cloud shape {cloud.shape} in {member.name}")
            out_dir = pc_root / cat
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / f"{obj}.npy.tmp"
            with open(tmp, "wb") as f:
                np.save(f, cloud.astype(np.float32))
            os.replace(tmp, out_dir / f"{obj}.npy")
            found[cat].append(obj)
    return found


# --------------------------------------------------------------------------- #
# state + local checks
# --------------------------------------------------------------------------- #
def load_state(out: Path) -> dict:
    p = out / STATE_NAME
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except ValueError:
            shutil.copy(p, p.with_name(STATE_NAME + ".corrupt"))
    return {"version": 1, "renders": {}, "point_clouds": {}}


def render_dir_complete(d: Path) -> bool:
    """Cheap filesystem check: every object folder has 24 PNGs and transforms.json."""
    if not d.is_dir():
        return False
    objs = [o for o in d.iterdir() if o.is_dir()]
    if not objs:
        return False
    for o in objs:
        files = [p.name for p in o.iterdir()]
        if sum(f.endswith(".png") for f in files) != VIEWS_PER_OBJECT or "transforms.json" not in files:
            return False
    return True


# --------------------------------------------------------------------------- #
# main flow
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resumable downloader for the remaining OmniObject3D categories "
        "(24-view renders + 4096-point clouds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=Path("dataset"))
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--categories", nargs="+", metavar="CAT", help="explicit list (default: every remote category not in our 76)")
    sel.add_argument("--all", action="store_true", help="every remote category, including our 76 (already-complete ones are adopted, not re-downloaded)")
    p.add_argument("--modality", choices=("both", "renders", "point_clouds"), default="both")
    p.add_argument("--workers", type=int, default=8, help="parallel connections per file")
    p.add_argument("--chunk-mb", type=int, default=8, help="resume granularity")
    p.add_argument("--max-retries", type=int, default=12, help="per chunk before giving up (0 = retry forever)")
    p.add_argument("--file-retries", type=int, default=2, help="whole-archive attempts after a checksum failure")
    p.add_argument("--status-interval", type=float, default=None, help="seconds between status updates (default 1 on a TTY, 30 otherwise)")
    p.add_argument("--delay", type=float, default=0.0, help="pause between archives (seconds)")
    p.add_argument("--force", action="store_true", help="re-download categories even if recorded as finished")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--no-space-check", action="store_true")
    p.add_argument("--use-fallback-keys", action="store_true", help="use the AK/SK published in the OmniObject3D README")
    p.add_argument("--dry-run", action="store_true", help="list the plan and exit; downloads nothing")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    console = Console(out / "download_remaining.log")
    want_r = args.modality in ("both", "renders")
    want_p = args.modality in ("both", "point_clouds")
    chunk = args.chunk_mb * MiB

    base.ensure_login(use_fallback_keys=args.use_fallback_keys)

    console.log("listing remote files...")
    renders = {Path(r.name).name[: -len(".tar.gz")]: r for r in list_remote(base.RENDER_PREFIX) if r.name.endswith(".tar.gz")}
    pc_path = base.PC_ARCHIVE_TMPL.format(point_count=4096).lstrip("/")
    pc_remote = next((r for r in list_remote("/" + pc_path) if r.name == pc_path), None)
    if want_p and pc_remote is None:
        console.log(f"ERROR: point-cloud archive {pc_path} not found on the remote")
        return 4

    if args.categories:
        cats = [resolve_category(c) for c in args.categories]
    elif args.all:
        cats = sorted(renders)
    else:
        cats = sorted(set(renders) - set(ALL_HOUSE_CATEGORIES))
    unknown = [c for c in cats if c not in renders]
    if unknown:
        console.log(f"ERROR: not on the remote: {', '.join(unknown)}")
        return 4

    state = load_state(out)
    state.setdefault("renders", {})
    state.setdefault("point_clouds", {})
    save_state = lambda: atomic_write_json(out / STATE_NAME, state)  # noqa: E731

    # ---- what is already finished?
    dl_dir, scratch = out / "_downloads", out / "_extract"
    done_r: list[str] = []
    todo_r: list[str] = []
    if want_r:
        for c in cats:
            rec = state["renders"].get(c)
            on_disk = (out / "renders" / c).is_dir()
            if not args.force and rec and on_disk:
                done_r.append(c)
            elif not args.force and not rec and render_dir_complete(out / "renders" / c):
                state["renders"][c] = {"adopted": True, "bytes": renders[c].size}
                done_r.append(c)  # complete copy already here (e.g. copied from another PC)
            else:
                todo_r.append(c)
        if not args.dry_run:
            save_state()
    done_p = [c for c in cats if c in state["point_clouds"] and (out / "point_clouds" / c).is_dir()] if want_p and not args.force else []
    todo_p = [c for c in cats if c not in done_p] if want_p else []
    todo_r.sort(key=lambda c: renders[c].size)  # small first: early wins, fast sanity check

    # ---- plan / dry-run
    plan_bytes = (sum(renders[c].size for c in cats) if want_r else 0) + (pc_remote.size if want_p else 0)
    pend_bytes = sum(renders[c].size for c in todo_r) + (pc_remote.size if todo_p else 0)
    largest = max([renders[c].size for c in todo_r] + [pc_remote.size if todo_p and pc_remote else 0] + [0])
    free = shutil.disk_usage(out).free
    need = int(pend_bytes * 1.05) + 2 * largest + GiB

    print(f"\nDataset        {base.DATASET_REPO}")
    print(f"Output dir     {out.resolve()}")
    print(f"Categories     {len(cats)} selected | renders: {len(done_r)} done, {len(todo_r)} to do"
          f" | point clouds: {len(done_p)} done, {len(todo_p)} to do")
    print(f"To download    {fmt_bytes(pend_bytes)} (renders {fmt_bytes(sum(renders[c].size for c in todo_r))}"
          f"{' + 4096-pt clouds ' + fmt_bytes(pc_remote.size) if todo_p else ''})")
    print(f"Disk           need ~{fmt_bytes(need)} incl. scratch; free {fmt_bytes(free)}")
    if args.dry_run:
        print(f"\n{'category':22s} {'archive':>10s}  status")
        for c in cats:
            if not want_r:
                break
            part = dl_dir / f"{c}.tar.gz.part.json"
            note = "done" if c in done_r else ("partial .part on disk" if part.exists() else "todo")
            print(f"{c:22s} {fmt_bytes(renders[c].size):>10s}  {note}")
        print("\nDry run - nothing downloaded.")
        return 0
    if not pend_bytes:
        console.log("Nothing to do - everything selected is already downloaded.")
        return 0
    if not args.no_space_check and free < need:
        console.log(f"ERROR: not enough free disk ({fmt_bytes(free)} < ~{fmt_bytes(need)}). Use --no-space-check to override.")
        return 5

    dl_dir.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)

    # ---- run
    stop = threading.Event()

    def on_sigint(signum, frame):  # noqa: ARG001
        if stop.is_set():
            console.log("second Ctrl+C - exiting immediately (state on disk is safe)")
            os._exit(130)
        stop.set()
        console.log("Ctrl+C - stopping gracefully; progress is saved. Re-run the same command to resume.")

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    initial_done = sum(renders[c].size for c in done_r) + (pc_remote.size if want_p and not todo_p else 0)
    n_items = len(done_r) + len(todo_r) + (1 if want_p else 0)
    prog = Progress(plan_bytes, initial_done, n_items, items_done=len(done_r) + (1 if want_p and not todo_p else 0))
    interval = args.status_interval or (1.0 if console.tty else 30.0)
    reporter = Reporter(prog, console, interval)
    reporter.start()

    failures: list[str] = []
    finished = 0
    t_start = time.monotonic()
    done_bytes_session = 0

    def fetch(remote: Remote, dest: Path, label: str) -> None:
        urls = UrlProvider(remote, console.log)
        for attempt in range(1, args.file_retries + 1):
            try:
                download_file(remote, dest, urls, prog, console, stop,
                              label=label, workers=args.workers, chunk=chunk, max_retries=args.max_retries)
                return
            except VerifyError as exc:
                console.log(f"  ! {exc} (attempt {attempt}/{args.file_retries})")
                if attempt == args.file_retries:
                    raise

    try:
        # 1) point clouds first: small, and lets the later per-category sanity check run
        if want_p and todo_p:
            label = "4096 point clouds"
            arch = dl_dir / Path(pc_remote.name).name
            t0 = time.monotonic()
            try:
                fetch(pc_remote, arch, label)
                prog.set_phase("extracting clouds", 0)
                found = extract_point_clouds(arch, set(todo_p), out / "point_clouds", prog)
                prog.end_file(True, file_bytes_in_total=pc_remote.size)
                for c, objs in found.items():
                    if not objs:
                        failures.append(f"point clouds {c}: not in archive")
                        continue
                    state["point_clouds"][c] = {"clouds": len(objs)}
                save_state()
                if not args.keep_archives:
                    arch.unlink(missing_ok=True)
                prog.item_finished()
                console.log(f"✔ point clouds: {sum(len(v) for v in found.values())} clouds for "
                            f"{sum(1 for v in found.values() if v)}/{len(todo_p)} categories "
                            f"({time.monotonic() - t0:.0f}s)")
            except (Stopped, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001
                prog.end_file(False, file_bytes_in_total=pc_remote.size)
                failures.append(f"point clouds: {exc}")
                console.log(f"✘ point clouds failed: {exc}")

        # 2) renders, smallest category first
        for k, cat in enumerate(todo_r, 1):
            if stop.is_set():
                raise Stopped()
            remote = renders[cat]
            arch = dl_dir / f"{cat}.tar.gz"
            t0 = time.monotonic()
            try:
                fetch(remote, arch, cat)
                prog.set_phase("extracting", 0)
                n_obj, n_img, warns = extract_renders(arch, cat, out / "renders", scratch, prog)
                prog.end_file(True, file_bytes_in_total=remote.size)
                state["renders"][cat] = {
                    "objects": n_obj, "images": n_img, "bytes": remote.size,
                    "sha256": remote.sha256, "warnings": warns,
                    "finished": dt.datetime.now().isoformat(timespec="seconds"),
                }
                save_state()
                if not args.keep_archives:
                    arch.unlink(missing_ok=True)
                prog.item_finished()
                finished += 1
                done_bytes_session += remote.size
                dur = time.monotonic() - t0
                console.log(f"✔ {cat}: {n_obj} objects / {n_img} images, {fmt_bytes(remote.size)} in "
                            f"{fmt_dur(dur)} ({fmt_rate(remote.size / dur if dur else None)} incl. verify+extract)")
                for w in warns:
                    console.log(f"    warning: {w}")
                console.log("   " + prog.render(None).split(" | now")[0], screen=False)
            except (Stopped, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001
                prog.end_file(False, file_bytes_in_total=remote.size)
                failures.append(f"{cat}: {exc}")
                prog.item_finished()
                console.log(f"✘ {cat} failed: {exc} - will be retried on the next run (partial data kept)")
            if args.delay > 0 and k < len(todo_r):
                time.sleep(args.delay)
    except (Stopped, KeyboardInterrupt):
        reporter.halt()
        console.log(f"Stopped after {finished} categories this session. Re-run the same command to resume.")
        return 130
    finally:
        reporter.halt()
        shutil.rmtree(scratch, ignore_errors=True)
        if not any(dl_dir.iterdir()):
            dl_dir.rmdir()

    # ---- consistency: every point-cloud set must match its render folders 1:1
    mism = []
    if want_r and want_p:
        for c in cats:
            rd, pd = out / "renders" / c, out / "point_clouds" / c
            if rd.is_dir() and pd.is_dir():
                a = {p.name for p in rd.iterdir() if p.is_dir()}
                b = {p.stem for p in pd.glob("*.npy")}
                if a != b:
                    mism.append(f"{c}: {len(a)} render objects vs {len(b)} clouds "
                                f"(only renders: {sorted(a - b)[:3]}, only clouds: {sorted(b - a)[:3]})")
    elapsed = time.monotonic() - t_start
    print()
    console.log(f"Session: {finished} categories, {fmt_bytes(done_bytes_session)} in {fmt_dur(elapsed)} "
                f"({fmt_rate(done_bytes_session / elapsed if elapsed else None)} average)")
    for m in mism:
        console.log(f"WARNING pairing mismatch - {m}")
    if failures:
        console.log(f"{len(failures)} failure(s) - re-run the same command to retry only those:")
        for f in failures:
            console.log(f"  - {f}")
        return 1
    console.log(f"All done. Dataset root: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
