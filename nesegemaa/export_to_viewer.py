#!/usr/bin/env python3
"""Export nesegemaa runs into the team 3D viewer (viewer/runs/<owner>__...).

Our evaluation stores predictions normalized to a unit ball (mean-centered,
divided by the max radius) — the frame the metrics were computed in. The
viewer draws every run in the GT's shared frame (bbox center, farthest GT
point = 1), so each prediction is mapped back through the GT object's own
normalization statistics and handed over with ``frame="raw"``:

    pred_raw_equiv = pred_normalized * gt_radius + gt_center

where gt_center = mean(raw GT) and gt_radius = max ||raw GT - gt_center||,
i.e. exactly the normalization our metrics used. No GT information enters the
prediction beyond aligning it the same way every run is aligned.

Supported inputs (all per-object ``pred.npy`` triplets):
- AtlasNet: ``results/qualitative/atlasnet/<cat>_<obj>/`` (written by atlas_qual)
- Point-E:  ``nesegemaa/pointe/triplets[_tag]/`` (written by faithful_eval)
- TripoSR:  ``nesegemaa/triposr/triplets[_tag]/`` (written by eval_triposr)

Usage:
    python nesegemaa/export_to_viewer.py --track atlasnet \
        --triplets results/qualitative/atlasnet --name a0_warmdec_postfix \
        --label "AtlasNet A0b warm-start (post-fix)" \
        --notes "5-object panel; normalized frame undone with GT stats"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viewer"))
from export import ViewerRun, load_gt_raw  # noqa: E402

MODELS = {"atlasnet": "AtlasNet", "point-e": "Point-E", "triposr": "TripoSR"}


def gt_normalization(obj_id: str):
    """Mean-center + max-radius stats, matching pp.normalize_unitL2ball."""
    gt = load_gt_raw(obj_id)
    center = gt.mean(axis=0)
    radius = float(np.linalg.norm(gt - center, axis=1).max())
    if radius < 1e-9:
        raise RuntimeError(f"degenerate GT for {obj_id}")
    return center, radius


def build_dataset_index():
    """Map '<cat>_<obj-stem>' (triplet dir naming) -> '<cat>/<obj-stem>'."""
    idx = {}
    for cat_dir in sorted((ROOT / "dataset" / "point_clouds").iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in cat_dir.glob("*.npy"):
            idx[f"{cat_dir.name}_{f.stem}"] = f"{cat_dir.name}/{f.stem}"
    return idx


DATASET_INDEX = build_dataset_index()


def resolve_object(d: Path) -> tuple[str, dict]:
    """Return (obj_id, metrics_or_empty). metrics.json when present, else parse dir name."""
    mp = d / "metrics.json"
    if mp.is_file():
        m = json.loads(mp.read_text())
        return f"{m['category']}/{m['object']}", m
    obj_id = DATASET_INDEX.get(d.name)
    if obj_id is None:
        raise RuntimeError(f"cannot resolve triplet dir {d.name} to a dataset object")
    return obj_id, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=sorted(MODELS), required=True)
    ap.add_argument("--triplets", type=Path, required=True)
    ap.add_argument("--owner", default="nesegemaa")
    ap.add_argument("--name", required=True, help="run id suffix, e.g. a0_warmdec_postfix")
    ap.add_argument("--label", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--limit", type=int, default=0, help="max objects (0 = all)")
    args = ap.parse_args()

    dirs = sorted(p for p in args.triplets.iterdir()
                  if p.is_dir() and (p / "pred.npy").is_file())
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        raise SystemExit(f"no pred.npy triplets under {args.triplets}")

    run = ViewerRun(owner=args.owner, model=MODELS[args.track], name=args.name,
                    label=args.label, notes=args.notes)
    n = 0
    for d in dirs:
        obj_id, m = resolve_object(d)
        pred = np.load(d / "pred.npy").astype(np.float64)
        center, radius = gt_normalization(obj_id)
        pred_raw = pred * radius + center  # undo our normalization -> raw frame
        info = {"view": m.get("view", 0)}
        if "chamfer" in m:
            info["panel_chamfer"] = round(float(m["chamfer"]), 5)
        if "fscore" in m:
            info["panel_fscore"] = round(float(m["fscore"]), 5)
        run.add(obj_id, pred_raw, frame="raw", info=info)
        n += 1
    out = run.save()
    print(f"exported {n} object(s) from {args.triplets} -> {out}")


if __name__ == "__main__":
    main()
