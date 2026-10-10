#!/usr/bin/env python3
"""Export the sparse COLMAP runs (results/colmap/sparse_{known,sfm}) to the team viewer.

COLMAP points live in the render frame; they are mapped back to the raw GT frame
with the inverse of the per-object fit in bones/gt_alignment.json, then the
viewer normalises them like every other run. Objects without a reliable fit are
shown but excluded from the scores.

    python bones/export_colmap_to_viewer.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "viewer"))
from export import ViewerRun  # noqa: E402
from gt_align import RELIABLE_INSIDE, ROOT, load_cameras  # noqa: E402

RUNS = {
    "known": ("sparse_known", "COLMAP, true cameras",
              "Sparse SIFT points triangulated with the 24 ground-truth cameras (multi-view best case)."),
    "sfm": ("sparse_sfm", "COLMAP, blind SfM",
            "Sparse incremental SfM without camera poses; registered views Sim(3)-aligned to the true cameras."),
}


def render_to_raw(P: np.ndarray, fit: dict) -> np.ndarray:
    """Inverse of gt_align.apply_alignment."""
    R = np.array(fit["rotation"])
    return ((np.asarray(P, np.float64) - np.array(fit["offset"])) / fit["scale"]) @ R + np.array(fit["center"])


def sfm_used(ws: Path, names) -> list[bool]:
    best = None
    for d in sorted((ws / "sparse").glob("*")):
        if (d / "images.bin").is_file():
            rec = pycolmap.Reconstruction(str(d))
            if best is None or rec.num_reg_images() > best.num_reg_images():
                best = rec
    reg = {im.name for im in best.images.values() if im.has_pose} if best else set()
    return [n in reg for n in names]


def main():
    align = json.loads((HERE / "gt_alignment.json").read_text())
    for mode, (name, label, notes) in RUNS.items():
        objects = json.loads((ROOT / f"results/colmap/{name}/objects.json").read_text())
        run = ViewerRun(owner="bones", model="COLMAP", name=name, label=label, notes=notes)
        for o in objects:
            obj, fit = o["object"], align.get(o["object"])
            if not fit or "rotation" not in fit:
                continue
            ws = HERE / "work" / mode / obj.replace("/", "__")
            pts = np.load(ws / "points.npy") if (ws / "points.npy").is_file() else np.zeros((0, 3))
            names, c2w, _, _ = load_cameras(ROOT / "dataset/renders" / obj)
            used = [True] * len(names) if mode == "known" else sfm_used(ws, names)
            excluded = None if fit["frac_inside"] >= RELIABLE_INSIDE else \
                f"GT fit unreliable ({fit['frac_inside']:.0%} of GT points inside the silhouettes)"
            run.add(obj, render_to_raw(pts, fit), frame="raw", info={"views used": f"{sum(used)}/24"},
                    cameras=render_to_raw(c2w[:, :3, 3], fit), cameras_used=used, excluded=excluded)
        run.save()


if __name__ == "__main__":
    main()
