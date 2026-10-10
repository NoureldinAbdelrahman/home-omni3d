#!/usr/bin/env python3
"""Score reconstructed point clouds against the OmniObject3D GT clouds.

Both clouds are in the render frame: the reconstruction natively, the GT via
the per-object fit in ``bones/gt_alignment.json`` (objects whose fit is
unreliable are reported but not scored). Both are then normalised with the
*GT's* unit sphere (GT bbox centre, max GT radius = 1), as in docs/METRICS.md.

Metrics (unit-sphere units):
* chamfer_l1 = mean NN dist pred->GT + mean NN dist GT->pred (docs/METRICS.md)
* precision / recall / F at tau = 1% (team metric), 2% and 5%
* f_ceiling_tau: F-score of the GT against itself (leave-one-out NN). The GT is
  only 4096 points (~1.6% spacing), so even a perfect surface cannot reach F=1
  at tau=1%; this is the sampling-limited ceiling for that object.

    python bones/eval_clouds.py results/colmap/sparse_known
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_align import RELIABLE_INSIDE, ROOT, apply_alignment  # noqa: E402

TAUS = (0.01, 0.02, 0.05)
ALIGNMENT = ROOT / "bones/gt_alignment.json"


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    c = (gt.max(0) + gt.min(0)) / 2
    r = np.linalg.norm(gt - c, axis=1).max()
    gt, pred = (gt - c) / r, (pred - c) / r
    loo = cKDTree(gt).query(gt, k=2)[0][:, 1]
    out = {f"f_ceiling_{int(t * 100)}": float((loo < t).mean()) for t in TAUS}
    out["n_pred"] = len(pred)
    if not len(pred):
        return out | {f"{k}_{int(t * 100)}": 0.0 for t in TAUS for k in ("precision", "recall", "f")}
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    out["chamfer_l1"] = float(d_pg.mean() + d_gp.mean())
    for t in TAUS:
        p, rc = float((d_pg < t).mean()), float((d_gp < t).mean())
        k = int(t * 100)
        out |= {f"precision_{k}": p, f"recall_{k}": rc, f"f_{k}": 2 * p * rc / (p + rc) if p + rc else 0.0}
    return out


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="dir with objects.json from colmap_sparse.py")
    ap.add_argument("--work", type=Path, default=ROOT / "bones/work")
    a = ap.parse_args()

    runs = json.loads((a.run_dir / "objects.json").read_text())
    align = json.loads(ALIGNMENT.read_text())
    rows = []
    for run in runs:
        obj, mode = run["object"], run["mode"]
        row = dict(object=obj, category=obj.split("/")[0], registered=run.get("registered"))
        info = align.get(obj, {})
        if "error" in run:
            row["status"] = "run_error"
        elif info.get("frac_inside", 0) < RELIABLE_INSIDE:
            row["status"] = "gt_unreliable"
        else:
            gt = apply_alignment(np.load(ROOT / "dataset/point_clouds" / f"{obj}.npy").astype(np.float64), info)
            pred = np.load(a.work / mode / obj.replace("/", "__") / "points.npy").astype(np.float64)
            row |= dict(status="ok", **score(pred, gt))
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.run_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    metric_keys = ["chamfer_l1"] + [f"{k}_{int(t * 100)}" for t in TAUS
                                    for k in ("precision", "recall", "f", "f_ceiling")]
    cats = sorted({r["category"] for r in ok})
    per_cat = {c: {k: mean([r for r in ok if r["category"] == c], k) for k in metric_keys + ["n_pred"]}
               for c in cats}
    summary = dict(
        mode=runs[0]["mode"] if runs else None, n_objects=len(rows), n_scored=len(ok),
        n_gt_unreliable=sum(r["status"] == "gt_unreliable" for r in rows),
        n_run_error=sum(r["status"] == "run_error" for r in rows),
        n_empty=sum(r.get("n_pred") == 0 for r in ok),
        registered_mean=mean(rows, "registered"),
        micro={k: mean(ok, k) for k in metric_keys + ["n_pred"]},
        macro={k: (float(np.mean([v[k] for v in per_cat.values() if v[k] is not None]))
                   if per_cat else None) for k in metric_keys},
        per_category=per_cat,
    )
    (a.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in summary if k != "per_category"}, indent=1))


if __name__ == "__main__":
    main()
