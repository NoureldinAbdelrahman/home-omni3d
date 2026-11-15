#!/usr/bin/env python3
"""Team 3D results viewer. Runs in the foreground; Ctrl+C stops it.

    python viewer/view.py                 # serve + open the browser
    python viewer/view.py --port 9000 --no-browser
    python viewer/view.py --export out/   # static copy (no server needed to host it)

Shows every run under viewer/runs/ (add yours with viewer/export.py) against the
ground truth. GT clouds and photos are read from dataset/ on demand.
"""
import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export import DATASET, QUANT, RUNS_DIR, gt_frame, load_gt_raw  # noqa: E402

THUMB, COLS = 192, 8
_lock = threading.Lock()


def load_runs() -> list[dict]:
    runs = []
    for p in sorted(RUNS_DIR.glob("*/run.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {p}: {exc}", file=sys.stderr)
    return runs


def category_objects(cat: str) -> list[str]:
    """Objects of ``cat`` that appear in any run, in a fixed order."""
    return sorted({o for r in load_runs() for o in r["objects"] if o.split("/")[0] == cat})


@lru_cache(maxsize=256)
def gt_payload(cat: str, ids: tuple) -> bytes:
    chunks, offsets, offset = [], {}, 0
    for obj_id in ids:
        try:
            gt = load_gt_raw(obj_id)
        except FileNotFoundError:
            offsets[obj_id] = [offset, 0]
            continue
        c, r = gt_frame(gt)
        q = np.clip(np.round((gt - c) / r / QUANT), -32767, 32767).astype("<i2")
        offsets[obj_id] = [offset, len(q)]
        chunks.append(q); offset += len(q)
    raw = np.concatenate(chunks).tobytes() if chunks else b""
    return json.dumps({"objects": list(ids), "offsets": offsets, "quant": QUANT, "thumb": THUMB, "cols": COLS,
                       "b64": base64.b64encode(raw).decode()}).encode()


@lru_cache(maxsize=256)
def photo_grid(cat: str, ids: tuple) -> bytes:
    rows = max(1, -(-len(ids) // COLS))
    grid = Image.new("RGBA", (THUMB * min(COLS, max(len(ids), 1)), THUMB * rows))
    for i, obj_id in enumerate(ids):
        p = DATASET / "renders" / obj_id / "000.png"
        if p.is_file():
            im = Image.open(p).convert("RGBA"); im.thumbnail((THUMB, THUMB))
            grid.paste(im, (THUMB * (i % COLS), THUMB * (i // COLS)))
    buf = io.BytesIO(); grid.save(buf, "WEBP", quality=82, method=4)
    return buf.getvalue()


def api(path: str):
    """(content-type, body) for /api/... paths, or None."""
    if path == "/api/runs.json":
        return "application/json", json.dumps(load_runs(), separators=(",", ":")).encode()
    for prefix, ext, fn, ctype in (("/api/gt/", ".json", gt_payload, "application/json"),
                                   ("/api/img/", ".webp", photo_grid, "image/webp")):
        if path.startswith(prefix) and path.endswith(ext):
            cat = path[len(prefix):-len(ext)]
            if "/" in cat or ".." in cat:
                return None
            with _lock:
                return ctype, fn(cat, tuple(category_objects(cat)))
    return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path.startswith("/api/"):
            res = api(path)
            if res is None:
                return self.send_error(404)
            ctype, body = res
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/runs/") and path.endswith("/run.json"):
            return self.send_error(404)  # served through /api/runs.json
        return super().do_GET()

    def log_message(self, fmt, *args):  # quiet: only errors
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def export_static(out: Path):
    runs = load_runs()
    if out.exists():
        shutil.rmtree(out)
    (out / "api/gt").mkdir(parents=True); (out / "api/img").mkdir(parents=True)
    shutil.copy2(HERE / "index.html", out / "index.html")
    (out / "api/runs.json").write_bytes(api("/api/runs.json")[1])
    cats = sorted({o.split("/")[0] for r in runs for o in r["objects"]})
    for cat in cats:
        (out / f"api/gt/{cat}.json").write_bytes(api(f"/api/gt/{cat}.json")[1])
        (out / f"api/img/{cat}.webp").write_bytes(api(f"/api/img/{cat}.webp")[1])
    for r in runs:
        shutil.copytree(RUNS_DIR / r["id"] / "pts", out / "runs" / r["id"] / "pts")
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"exported {len(runs)} runs, {len(cats)} categories, {n} files -> {out}")


def open_browser(url: str):
    try:
        if "microsoft" in Path("/proc/version").read_text().lower() and shutil.which("explorer.exe"):
            subprocess.Popen(["explorer.exe", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except OSError:
        pass
    webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--export", type=Path, metavar="DIR", help="write a static copy instead of serving")
    a = ap.parse_args()

    if a.export:
        return export_static(a.export)
    if not (DATASET / "point_clouds").is_dir():
        print(f"note: {DATASET} not found - runs will show without ground truth or photos", file=sys.stderr)
    runs = load_runs()
    for port in range(a.port, a.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit(f"no free port in {a.port}-{a.port + 19}")
    url = f"http://localhost:{port}/"
    print(f"Viewer: {url}   ({len(runs)} runs: {', '.join(r['id'] for r in runs) or 'none yet'})")
    print("Press Ctrl+C to stop.", flush=True)
    if not a.no_browser and not os.environ.get("SSH_CONNECTION"):
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
