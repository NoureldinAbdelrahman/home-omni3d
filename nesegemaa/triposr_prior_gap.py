#!/usr/bin/env python3
"""Quantify TripoSR's deviation from OmniObject3D GT: pose error or shape error?

TripoSR's outputs look clean but score worse than AtlasNet's on Chamfer,
suggesting over-reliance on pre-trained priors. This script separates two
hypotheses on the 5 shared panel objects:

  H1 (frame mismatch): TripoSR reconstructs the right shape in the wrong
       rigid frame (its Objaverse-era canonical pose vs our GT frame).
       Then rigid ICP alignment should collapse most of the Chamfer gap.
  H2 (prior overfitting): the shape itself is a generic prior sample that
       ignores input details. Then even optimally-aligned Chamfer stays high.

Control: the AtlasNet winner was *trained* to output our GT frame, so its
aligned-vs-raw gap should be ~zero. AtlasNet preds are subsampled to 4096
(seed 0) to match TripoSR/Point-E density (see PAPER audit note).

Method: point-to-point ICP (25 iters, Kabsch via SVD, scipy KDTree only),
scoring in the same squared-distance units / tau=0.001 as the eval tracks.

Usage:
    python nesegemaa/triposr_prior_gap.py
Writes nesegemaa/triposr_prior_gap.csv and prints the table.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
NES = ROOT / "nesegemaa"
TAU_SQ = 0.001
PANEL = ["cup_cup_003", "hammer_hammer_011",
         "medicine_bottle_medicine_bottle_068",
         "shampoo_shampoo_003", "teapot_teapot_003"]


def score(pred, gt, tau_sq=TAU_SQ):
    da, _ = cKDTree(gt).query(pred)
    db, _ = cKDTree(pred).query(gt)
    da2, db2 = da ** 2, db ** 2
    chamfer = float(da2.mean() + db2.mean())
    p = float((db2 < tau_sq).mean())
    r = float((da2 < tau_sq).mean())
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return chamfer, f


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R, cd - R @ cs


def icp_align(src, dst, iters=25):
    """Rigid-align src onto dst. Returns aligned src (same shape)."""
    cur = src.copy()
    for _ in range(iters):
        _, idx = cKDTree(dst).query(cur)
        R, t = kabsch(cur, dst[idx])
        cur = cur @ R.T + t
    return cur


def load_panel(track_dir, n_atlas=None):
    out = {}
    for d in PANEL:
        pred = np.load(track_dir / d / "pred.npy").astype(np.float64)
        gt = np.load(track_dir / d / "gt.npy").astype(np.float64)
        if n_atlas is not None and len(pred) > n_atlas:
            rng = np.random.default_rng(0)
            pred = pred[rng.choice(len(pred), n_atlas, replace=False)]
        out[d] = (pred, gt)
    return out


def main():
    tri = load_panel(NES / "triposr" / "triplets")
    atl = load_panel(ROOT / "results" / "qualitative" / "atlasnet"
                     / "a2_p25_sphere_s123", n_atlas=4096)
    rows = []
    for d in PANEL:
        for name, store in (("triposr", tri), ("atlasnet_s123", atl)):
            pred, gt = store[d]
            raw_ch, raw_f = score(pred, gt)
            al = icp_align(pred, gt)
            al_ch, al_f = score(al, gt)
            rows.append({"object": d, "model": name,
                         "raw_chamfer": round(raw_ch, 4),
                         "aligned_chamfer": round(al_ch, 4),
                         "gap": round(raw_ch - al_ch, 4),
                         "raw_f": round(raw_f, 4),
                         "aligned_f": round(al_f, 4)})
    with open(NES / "triposr_prior_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'object':32s} {'model':13s} {'raw_ch':>7s} {'align_ch':>8s} {'gap':>7s} {'raw_f':>6s} {'align_f':>7s}")
    for r in rows:
        print(f"{r['object']:32s} {r['model']:13s} {r['raw_chamfer']:7.4f} "
              f"{r['aligned_chamfer']:8.4f} {r['gap']:7.4f} {r['raw_f']:6.4f} {r['aligned_f']:7.4f}")
    print("wrote nesegemaa/triposr_prior_gap.csv")


if __name__ == "__main__":
    main()
