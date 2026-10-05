#!/usr/bin/env python3
"""Faithful Point-E image-to-3D evaluation (nesegemaa track).

Two-stage pipeline exactly as released by OpenAI, no substitutions:

- Stage 1 ``base40M-imagevec``: CLIP ViT-L/14 image embedding (computed inside
  point-e from the PIL image) conditions a 40M diffusion model -> 1024 points.
- Stage 2 ``upsample``: conditioned on the 1024-point cloud -> +3072 points
  (total 4096) with RGB. ``--points 1024`` runs stage 1 only.

Rules honored (see branch scope):
- Reconstruction input is the IMAGE ONLY. GT point clouds are loaded solely
  for scoring (Chamfer + F-score@0.01, same helper as the AtlasNet track).
- Objects must come from the ``splits.json`` TEST lists (else the category's
  ``test[0]`` is used and the substitution is printed).
- UnitBall normalization identical in spirit to the AtlasNet eval: subtract
  the mean, divide by the max radius (documented in the summary).
- Fixed seeds per (object, seed) for determinism. Diffusion is stochastic:
  Phase-2 sweeps report mean+-std over seeds.
- NO fallback of any kind: any failure (weights, CLIP, sampling, scoring)
  raises. A ``fallback:none`` flag is written into every summary.

Outputs (default names; ``--tag X`` writes ``results_X.csv``/``summary_X.json``):
``nesegemaa/pointe/results.csv`` (one row per object x seed) and
``nesegemaa/pointe/summary.json`` (config + weight provenance + aggregates
with micro/macro and <3-object noise flags).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (script lives in nesegemaa/pointe/)
ATL = ROOT / "AtlasNet"
sys.path.insert(0, str(ATL))
sys.path.insert(0, str(ROOT))

import dataset.pointcloud_processor as pp  # noqa: E402

TAU_SQ = 0.001  # squared distances, same as AtlasNet training/eval
TAU_EUCLID_APPROX = TAU_SQ ** 0.5  # ~= 0.0316
DEFAULT_PANEL = [
    ("medicine_bottle", "medicine_bottle_068"),
    ("cup", "cup_032"),
    ("teapot", "teapot_031"),
    ("hammer", "hammer_019"),
    ("shampoo", "shampoo_031"),
]
CLIP_MODEL = "ViT-L/14"


def chamfer_and_fscore(a, b, tau_sq=TAU_SQ):
    # Squared-distance units, identical to the AtlasNet track definition.
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    da2, db2 = da ** 2, db ** 2
    chamfer = float(da2.mean() + db2.mean())
    precision = float((db2 < tau_sq).mean())
    recall = float((da2 < tau_sq).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return chamfer, fscore


def unit_ball(pts):
    """Center + scale to unit sphere. Formula recorded in every summary."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c, axis=1).max()
    if r < 1e-9:
        raise RuntimeError("degenerate point cloud (zero radius)")
    return ((pts - c) / r).astype(np.float32)


def denoise_cloud(pts, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal (P4 ablation). Returns filtered (M,3)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts)))
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                              std_ratio=std_ratio)
    out = np.asarray(clean.points, dtype=np.float32)
    if len(out) < 100:
        raise RuntimeError(f"denoise removed too much: {len(out)} points left")
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_provenance(cache_dir):
    """URLs + SHAs of every .pt in the point-e/CLIP cache. No silent anything."""
    from point_e.models.download import MODEL_PATHS
    rows = []
    for pt in sorted(Path(cache_dir).rglob("*.pt")):
        rows.append({"file": pt.name, "bytes": pt.stat().st_size,
                     "sha256": sha256_file(pt)})
    return {"model_urls": dict(MODEL_PATHS), "clip_model": CLIP_MODEL,
            "cache_files": rows, "backbone": "frozen", "fallback": "none"}


def load_image(path, preprocessing):
    im = Image.open(path)
    if preprocessing == "raw":
        return im.convert("RGB")
    if preprocessing == "masked":
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert("RGB")
    if preprocessing == "crop":
        from plotting import load_display_image
        return load_display_image(path).convert("RGB")
    raise ValueError(f"unknown preprocessing {preprocessing!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objects", nargs="+", default=None,
                    help="'cat:obj' pairs; default: the 5 MS1 panel objects")
    ap.add_argument("--view", type=int, default=0,
                    help="render index to condition on (prepared layout: 00..23)")
    ap.add_argument("--views", default=None,
                    help="comma-separated render indices for multi-view fusion "
                         "(overrides --view; needs --fusion)")
    ap.add_argument("--fusion", choices=["first", "best", "median"], default="first",
                    help="first=use views[0] (default, identical to --view); "
                         "best=per-object min-chamfer view; "
                         "median=coordinate-wise median cloud across views")
    ap.add_argument("--denoise", action="store_true",
                    help="statistical outlier removal (open3d, nb=20, std=2.0) "
                         "before scoring; ablated on/off in P4")
    ap.add_argument("--preprocessing", choices=["raw", "masked", "crop"], default="masked")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--points", type=int, choices=[1024, 4096], default=4096)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--tag", default="", help="suffix for results_<tag>.csv/summary_<tag>.json")
    ap.add_argument("--outdir", type=Path, default=ROOT / "nesegemaa" / "pointe")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    assert seeds, "--seeds must list at least one seed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Point-E eval requires CUDA on this box; refusing CPU fallback")

    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import default_cache_dir, load_checkpoint

    def load_stage(name):
        model = model_from_config(MODEL_CONFIGS[name], device)
        model.eval()
        model.load_state_dict(load_checkpoint(name, device))
        return model, diffusion_from_config(DIFFUSION_CONFIGS[name])

    print(f"[pointe] loading base40M-imagevec (guidance={args.guidance}) ...", flush=True)
    base_model, base_diff = load_stage("base40M-imagevec")
    up_model = up_diff = None
    stages = [(base_model, base_diff, 1024, args.guidance)]
    if args.points == 4096:
        print("[pointe] loading upsample ...", flush=True)
        up_model, up_diff = load_stage("upsample")
        stages.append((up_model, up_diff, 4096 - 1024, 1.0))
    n_stages = len(stages)
    sampler = PointCloudSampler(
        device=device,
        models=[s[0] for s in stages],
        diffusions=[s[1] for s in stages],
        num_points=[s[2] for s in stages],
        aux_channels=["R", "G", "B"],
        guidance_scale=[s[3] for s in stages],
        # Explicit per-stage sequences: the sampler only auto-expands them
        # for multi-stage setups, so single-stage (1024-pt) calls must pass
        # length-1 lists or the constructor asserts. Values = point-e defaults.
        use_karras=[True] * n_stages,
        karras_steps=[64] * n_stages,
        sigma_min=[1e-3] * n_stages,
        sigma_max=([120, 160])[:n_stages],
        s_churn=([3, 0])[:n_stages],
    )
    cache_dir = str(default_cache_dir())
    prov = weight_provenance(cache_dir)
    print(f"[pointe] provenance: {len(prov['cache_files'])} cached .pt files, "
          f"backbone={prov['backbone']} fallback={prov['fallback']}")

    splits = json.loads((ATL / "dataset" / "data" / "splits.json").read_text())
    test_lists = {c: sp.get("test", []) for c, sp in splits.items()}
    pairs = []
    for item in (args.objects or [f"{c}:{o}" for c, o in DEFAULT_PANEL]):
        c, o = item.split(":")
        if o in test_lists.get(c, []):
            pairs.append((c, o))
        else:
            fb = test_lists.get(c, [None])[0]
            print(f"  note: {c}:{o} not in test split -> using {c}:{fb}")
            if fb is not None:
                pairs.append((c, fb))
    assert pairs, "no test objects selected"

    rows = []
    suffix = f"_{args.tag}" if args.tag else ""
    trip = args.outdir / f"triplets{suffix}"
    trip.mkdir(parents=True, exist_ok=True)
    views = [int(v) for v in (args.views.split(",") if args.views else [args.view])]
    assert views, "no views selected"
    if args.fusion == "first":
        views = views[:1]
    for cat, obj in pairs:
        gt_raw = np.load(ATL / "dataset" / "data" / "ShapeNetV1PointCloud"
                         / cat / f"{obj}.npy").astype(np.float64)[:, :3]
        gt = unit_ball(gt_raw)
        for seed in seeds:
            per_view = []
            for v in views:
                img_path = (ATL / "dataset" / "data" / "ShapeNetV1Renderings"
                            / cat / obj / "rendering" / f"{v:02d}.png")
                if not img_path.is_file():
                    raise RuntimeError(f"render missing: {img_path}")
                pil = load_image(img_path, args.preprocessing)
                torch.manual_seed(seed)
                np.random.seed(seed)
                t0 = time.time()
                samples = None
                for x in sampler.sample_batch_progressive(
                        batch_size=1, model_kwargs=dict(images=[pil])):
                    samples = x
                pc = sampler.output_to_point_clouds(samples)[0]
                pred = unit_ball(pc.coords)
                if args.denoise:
                    pred = denoise_cloud(pred)
                dt = time.time() - t0
                ch, fs = chamfer_and_fscore(pred, gt, tau_sq=TAU_SQ)
                per_view.append((v, pred, ch, fs, dt))
            if args.fusion == "best":
                v, pred, ch, fs, dt = min(per_view, key=lambda t: t[2])
                fused = f"best-of-{len(views)}"
            elif args.fusion == "median" and len(per_view) > 1:
                stacked = np.stack([p for _, p, _, _, _ in per_view], axis=0)
                pred = np.median(stacked, axis=0).astype(np.float32)
                ch, fs = chamfer_and_fscore(pred, gt, tau_sq=TAU_SQ)
                dt = sum(t[4] for t in per_view)
                v, fused = -1, f"median-of-{len(views)}"
            else:
                v, pred, ch, fs, dt = per_view[0]
                fused = "first" if len(per_view) == 1 else "first"
                v = per_view[0][0]
            rows.append({
                "object": f"{cat}_{obj}", "category": cat, "view": v,
                "fusion": fused,
                "preprocessing": args.preprocessing, "guidance": args.guidance,
                "points": args.points, "seed": seed,
                "denoise": bool(args.denoise),
                "chamfer": ch, "fscore": round(fs, 6),
                "seconds": round(dt, 1),
            })
            td = trip / f"{cat}_{obj}"
            td.mkdir(parents=True, exist_ok=True)
            np.save(td / "pred.npy", np.asarray(pred, dtype=np.float32))
            np.save(td / "gt.npy", np.asarray(gt, dtype=np.float32))
            pil.save(td / "input.png")
            print(f"  {cat:16s} {obj:22s} seed={seed} {fused:12s} chamfer={ch:.4f} "
                  f"fscore={fs:.4f} ({dt:.0f}s)", flush=True)

    # aggregate: micro over objects, macro over categories, noise flags
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    cat_rows = [(c, float(np.mean([r["chamfer"] for r in rs])),
                 float(np.mean([r["fscore"] for r in rs])), len({r['object'] for r in rs}))
                for c, rs in sorted(cats.items())]
    noise = sorted(c for c, _, _, n in cat_rows if n < 3)
    summary = {
        "method": "Point-E faithful (base40M-imagevec -> upsample)",
        "config": {"views": views, "fusion": args.fusion,
                   "preprocessing": args.preprocessing,
                   "guidance": args.guidance, "points": args.points, "seeds": seeds,
                   "denoise": bool(args.denoise),
                   "tau_squared": TAU_SQ, "tau_euclid_approx": round(TAU_EUCLID_APPROX, 4),
                   "normalization": "subtract mean, divide by max radius (unit sphere)"},
        "weights": prov,
        "n_rows": len(rows), "n_objects": len({r["object"] for r in rows}),
        "micro_chamfer": float(np.mean([r["chamfer"] for r in rows])),
        "micro_fscore": float(np.mean([r["fscore"] for r in rows])),
        "macro_chamfer": float(np.mean([m[1] for m in cat_rows])) if cat_rows else None,
        "macro_fscore": float(np.mean([m[2] for m in cat_rows])) if cat_rows else None,
        "noise_categories_lt3": noise,
        "per_category": [{"category": c, "n": n, "chamfer": ch, "fscore": fs}
                         for c, ch, fs, n in cat_rows],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.outdir / f"results{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (args.outdir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("micro chamfer=%.4f fscore=%.4f | macro chamfer=%.4f fscore=%.4f | noise=%s"
          % (summary["micro_chamfer"], summary["micro_fscore"],
             summary["macro_chamfer"], summary["macro_fscore"], noise or "-"))
    print("wrote", args.outdir / f"results{suffix}.csv",
          "and", args.outdir / f"summary{suffix}.json")


if __name__ == "__main__":
    main()
