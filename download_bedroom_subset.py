#!/usr/bin/env python3
"""Targeted downloader for the bedroom subset of OmniObject3D (OpenDataLab).

Fetches only ``blender_renders_24_views`` images and per-category point-cloud
HDF5 files for the categories in :mod:`bedroom_categories`, extracts them, and
reorganizes everything into a PyTorch-friendly layout::

    dataset/
    ├── point_clouds/
    │   ├── bed/
    │   │   ├── bed_001.npy
    │   │   └── ...
    │   └── light/
    └── renders/
        ├── bed/
        │   ├── bed_001/
        │   │   ├── 000.png ... 023.png
        │   │   └── transforms.json
        │   └── ...
        └── light/

Usage::

    export OPENXLAB_AK=... OPENXLAB_SK=...
    python download_bedroom_subset.py --output-dir dataset --point-count 4096

    python download_bedroom_subset.py --dry-run
    python download_bedroom_subset.py --categories bed pillow lamp
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from bedroom_categories import BEDROOM_CATEGORIES, resolve_category

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("download_bedroom_subset")

DATASET_REPO = "omniobject3d/OmniObject3D-New"
LOCAL_REPO_DIR = DATASET_REPO.replace("/", "___")
RENDER_PREFIX = "/raw/blender_renders_24_views/img"
PC_PREFIX_TMPL = "/raw/point_clouds/hdf5_files/{point_count}"
DEFAULT_FALLBACK_AK = "bmyqk5wpbaxl6x1vkzq9"
DEFAULT_FALLBACK_SK = "nl7kq9palyr6j3pwxolden7ezq4dwjmbgdm81yeo"


class AuthError(RuntimeError):
    """Raised when OpenDataLab credentials are missing or rejected."""


class DownloadError(RuntimeError):
    """Raised when a remote file cannot be fetched or verified."""


@dataclass(frozen=True)
class RemoteFile:
    path: str  # e.g. /raw/blender_renders_24_views/img/bed.tar.gz
    size: int
    sha256: str = ""

    @property
    def name(self) -> str:
        return self.path.lstrip("/")


def ensure_login(use_fallback_keys: bool = False) -> None:
    """Authenticate via env vars, cached config, or the published fallback keys."""
    import openxlab

    ak = os.environ.get("OPENXLAB_AK", "").strip()
    sk = os.environ.get("OPENXLAB_SK", "").strip()

    if ak and sk:
        openxlab.login(ak=ak, sk=sk)
        return

    if use_fallback_keys:
        openxlab.login(ak=DEFAULT_FALLBACK_AK, sk=DEFAULT_FALLBACK_SK)
        return

    config_path = Path.home() / ".openxlab" / "config.json"
    if config_path.is_file():
        try:
            list_remote_files("/README.md")
            return
        except (AuthError, DownloadError, SystemExit, Exception):  # noqa: BLE001
            pass

    raise AuthError(
        "OpenDataLab credentials not found. Set OPENXLAB_AK / OPENXLAB_SK, "
        "run `openxlab login`, or pass --use-fallback-keys."
    )


def _client():
    from openxlab.dataset.commands.utility import ContextInfoNoLogin

    return ContextInfoNoLogin().get_client()


def list_remote_files(prefix: str) -> list[RemoteFile]:
    """Paginated remote listing restricted to ``prefix``."""
    try:
        api = _client().get_api()
        dataset_name = DATASET_REPO.replace("/", ",")
        files: list[RemoteFile] = []
        after: str | None = None
        while True:
            page = api.get_dataset_files(
                dataset_name=dataset_name,
                payload={"prefix": prefix},
                needContent=True,
                after=after,
                limit=500,
            )
            for item in page.get("list", []):
                path = item["path"]
                if path.startswith("//"):
                    path = path[1:]
                files.append(
                    RemoteFile(
                        path=path,
                        size=int(item.get("size", 0)),
                        sha256=item.get("sha256") or "",
                    )
                )
            if not page.get("hasNext", False):
                break
            after = page.get("after")
            if not after:
                break
    except SystemExit as exc:
        raise AuthError(f"Failed to list remote files under {prefix!r}") from exc
    except AuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AuthError(f"Failed to list remote files under {prefix!r}: {exc}") from exc

    if not files:
        raise DownloadError(f"No remote files found under {prefix!r}")
    return files


def _sdk_download(source_path: str, target_path: Path) -> Path:
    """Download one remote path via the openxlab SDK (native progress display)."""
    from openxlab.dataset import download as odl_download

    target_path.mkdir(parents=True, exist_ok=True)
    odl_download(
        dataset_repo=DATASET_REPO,
        source_path=source_path,
        target_path=str(target_path),
    )
    local = target_path / LOCAL_REPO_DIR / source_path.lstrip("/")
    if not local.is_file():
        raise DownloadError(f"SDK reported success but file missing: {local}")
    return local


def _extract_renders(archive: Path, category: str, renders_root: Path) -> list[str]:
    """Extract a category render tarball into ``renders/<category>/``."""
    category_dir = renders_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    object_ids: list[str] = []
    extract_dir = archive.parent / f"_extract_{category}"

    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(path=extract_dir, filter="data")
        except TypeError:  # filter= added in 3.12, backported to some 3.10.x
            tar.extractall(path=extract_dir)

    try:
        for entry in sorted(extract_dir.iterdir()):
            if not entry.is_dir():
                continue
            object_ids.append(entry.name)
            target = category_dir / entry.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(entry), str(target))
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
    return object_ids


def _extract_point_clouds(
    hdf5_path: Path,
    category: str,
    point_cloud_root: Path,
    object_ids: Sequence[str],
) -> int:
    """Split a per-category HDF5 ``(n_obj, n_points, 3)`` into per-object .npy files."""
    out_dir = point_cloud_root / category
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise DownloadError(f"Unexpected HDF5 layout in {hdf5_path}: missing 'data'")
        data = np.asarray(f["data"], dtype=np.float32)

    if data.ndim != 3 or data.shape[-1] != 3:
        raise DownloadError(
            f"Unexpected HDF5 shape {data.shape} in {hdf5_path}; expected (n, N, 3)"
        )

    n_objects = data.shape[0]
    ids = list(object_ids)
    if len(ids) != n_objects:
        if len(ids) < n_objects:
            ids = ids + [f"{category}_{i:03d}" for i in range(len(ids), n_objects)]
        else:
            logger.warning(
                "%s: %d render objects but %d point clouds; extras skipped",
                category,
                len(ids),
                n_objects,
            )
            ids = ids[:n_objects]

    for object_id, cloud in zip(ids, data):
        np.save(out_dir / f"{object_id}.npy", cloud)
    return n_objects


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.2f}TB"


def plan_downloads(
    categories: Sequence[str],
    point_count: int,
    want_renders: bool,
    want_point_clouds: bool,
) -> list[tuple[str, str, RemoteFile]]:
    """Return ``(kind, category, RemoteFile)`` rows for the requested subset."""
    plan: list[tuple[str, str, RemoteFile]] = []

    if want_renders:
        by_path = {f.path: f for f in list_remote_files(RENDER_PREFIX)}
        for cat in categories:
            path = f"{RENDER_PREFIX}/{cat}.tar.gz"
            if path not in by_path:
                sample = ", ".join(sorted(Path(p).stem for p in by_path)[:8])
                raise DownloadError(
                    f"Category {cat!r} has no render archive at {path} "
                    f"(examples: {sample})"
                )
            plan.append(("render", cat, by_path[path]))

    if want_point_clouds:
        prefix = PC_PREFIX_TMPL.format(point_count=point_count)
        by_path = {f.path: f for f in list_remote_files(prefix)}
        for cat in categories:
            # Normal layout is /.../cat_N.hdf5; a few entries nest an extra dir.
            matches = [
                f
                for p, f in by_path.items()
                if Path(p).name in {f"{cat}_{point_count}.hdf5", f"{cat}.hdf5"}
                or p.rstrip("/").endswith(f"/{cat}_{point_count}.hdf5")
            ]
            if not matches:
                raise DownloadError(
                    f"Category {cat!r} has no point cloud file under {prefix}"
                )
            plan.append(("point_cloud", cat, matches[0]))

    return plan


def run(args: argparse.Namespace) -> int:
    categories = (
        [resolve_category(c) for c in args.categories]
        if args.categories
        else list(BEDROOM_CATEGORIES)
    )
    want_renders = args.modality in ("both", "renders")
    want_point_clouds = args.modality in ("both", "point_clouds")

    if not want_renders and not want_point_clouds:
        logger.error("Nothing selected to download")
        return 2

    ensure_login(use_fallback_keys=args.use_fallback_keys)

    plan = plan_downloads(categories, args.point_count, want_renders, want_point_clouds)
    total_bytes = sum(r.size for _, _, r in plan)

    print(f"Dataset:  {DATASET_REPO}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Files:    {len(plan)}  |  Total: {_human_bytes(total_bytes)}")
    for kind, cat, remote in plan:
        print(f"  [{kind:11s}] {cat:14s} {remote.path}  ({_human_bytes(remote.size)})")

    if args.dry_run:
        print("Dry run — nothing downloaded.")
        return 0

    output_dir: Path = args.output_dir
    download_dir = output_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "renders").mkdir(parents=True, exist_ok=True)
    (output_dir / "point_clouds").mkdir(parents=True, exist_ok=True)

    object_ids_by_cat: dict[str, list[str]] = {}
    failures: list[str] = []

    for idx, (kind, cat, remote) in enumerate(plan, start=1):
        print(f"[{idx}/{len(plan)}] {kind} · {cat} · {remote.path}")
        try:
            local = _sdk_download(remote.path, download_dir)
        except SystemExit as exc:
            failures.append(f"{remote.path}: openxlab exited with {exc}")
            logger.error("download failed: %s", remote.path)
            continue
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            failures.append(f"{remote.path}: {exc}")
            logger.error("download failed: %s", exc)
            continue

        if kind == "render":
            try:
                object_ids_by_cat[cat] = _extract_renders(
                    local, cat, output_dir / "renders"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"extract renders {cat}: {exc}")
                logger.error("extract failed for %s: %s", cat, exc)
            if not args.keep_archives:
                local.unlink(missing_ok=True)
        else:
            try:
                n = _extract_point_clouds(
                    local,
                    cat,
                    output_dir / "point_clouds",
                    object_ids_by_cat.get(cat, []),
                )
                print(f"  → point_clouds/{cat}: {n} objects")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"extract point clouds {cat}: {exc}")
                logger.error("point cloud extract failed for %s: %s", cat, exc)
            if not args.keep_archives:
                local.unlink(missing_ok=True)

        if args.delay > 0 and idx < len(plan):
            time.sleep(args.delay)

    shutil.rmtree(download_dir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"\nDone. Dataset root: {output_dir.resolve()}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download the bedroom subset of OmniObject3D from OpenDataLab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Local dataset root (creates renders/ and point_clouds/).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        default=None,
        help="Override category list (aliases: lamp, wardrobe, nightstand).",
    )
    p.add_argument(
        "--modality",
        choices=("both", "renders", "point_clouds"),
        default="both",
        help="Which modalities to download.",
    )
    p.add_argument(
        "--point-count",
        type=int,
        choices=(1024, 4096, 16384),
        default=4096,
        help="Point-cloud resolution for HDF5 files.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between remote file downloads (rate limit).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan and exit.",
    )
    p.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar.gz / .hdf5 archives after extraction.",
    )
    p.add_argument(
        "--use-fallback-keys",
        action="store_true",
        help="Use the AK/SK published in the official OmniObject3D README.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run(args)
    except AuthError as exc:
        print(f"Auth error: {exc}", file=sys.stderr)
        return 3
    except DownloadError as exc:
        print(f"Download error: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
