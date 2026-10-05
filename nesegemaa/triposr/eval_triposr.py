#!/usr/bin/env python3
"""Faithful TripoSR image-to-3D evaluation (nesegemaa track).

Feed-forward pipeline exactly as released (no substitutions except the
documented CPU marching-cubes shim, which only affects device):

- weights: ``stabilityai/TripoSR`` (``config.yaml`` + ``model.ckpt``),
  SHAs recorded in every summary (backbone: frozen).
- input: single PIL image (view 0 default); preprocessing variants mirror the
  Point-E track: ``raw`` (RGB as-is), ``masked`` (RGBA over white), ``crop``
  (tight square crop). TripoSR's own rembg step is NOT used (our renders
  carry alpha; adding it would inject an undocumented prior).
- ``model([image])`` -> scene codes -> ``extract_mesh`` (resolution 256) ->
  ``trimesh.sample_surface`` for N points.
- UnitBall normalization identical to the other tracks (subtract mean,
  divide by max radius); scoring in squared-distance units, tau 0.001.
- TripoSR is deterministic (no diffusion): one pass per object, seed 0.
  GT clouds are scoring-only, never fed to the model.

Outputs (``--tag X`` writes ``results_X.csv``/``summary_X.json``):
``nesegemaa/triposr/results.csv`` (one row per object) and
``nesegemaa/triposr/summary.json`` (config + weight provenance + aggregates
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

TRIPOSR_SRC = "/media/susan/429428ec-710b-483c-9aaa-d0c4b6968baa/home-omni3d/thirdparty/TripoSR"
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (script lives in nesegemaa/triposr/)
ATL = ROOT / "AtlasNet"
sys.path.insert(0, TRIPOSR_SRC)
sys.path.insert(0, str(ATL))
sys.path.insert(0, str(ROOT))

TAU_SQ = 0.001  # squared distances, same as AtlasNet training/eval
TAU_EUCLID_APPROX = TAU_SQ ** 0.5
DEFAULT_PANEL = [
    ("medicine_bottle", "medicine_bottle_068"),
    ("cup", "cup_032"),
    ("teapot", "teapot_031"),
    ("hammer", "hammer_019"),
    ("shampoo", "shampoo_031"),
]


def chamfer_and_fscore(a, b, tau_sq=TAU_SQ):
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
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c, axis=1).max()
    if r < 1e-9:
        raise RuntimeError("degenerate point cloud (zero radius)")
    return ((pts - c) / r).astype(np.float32)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--preprocessing", choices=["raw", "masked", "crop"], default="masked")
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--mc-resolution", type=int, default=128)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", type=Path, default=ROOT / "nesegemaa" / "triposr")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TripoSR eval requires CUDA on this box; refusing CPU fallback")

    import trimesh
    from tsr.system import TSR
    from huggingface_hub import snapshot_download

    print("[triposr] loading stabilityai/TripoSR ...", flush=True)
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.to(device)
    model.eval()
    if hasattr(model, "renderer") and hasattr(model.renderer, "set_chunk_size"):
        model.renderer.set_chunk_size(2048)  # fit 11 GB alongside training

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

    try:
        import torchmcubes
        mc_backend = getattr(torchmcubes, "__backend__", "tatsy-cuda")
    except ImportError:
        mc_backend = "unavailable"

    rows = []
    suffix = f"_{args.tag}" if args.tag else ""
    trip = args.outdir / f"triplets{suffix}"
    trip.mkdir(parents=True, exist_ok=True)
    for cat, obj in pairs:
        img_path = (ATL / "dataset" / "data" / "ShapeNetV1Renderings"
                    / cat / obj / "rendering" / f"{args.view:02d}.png")
        if not img_path.is_file():
            raise RuntimeError(f"render missing: {img_path}")
        pil = load_image(img_path, args.preprocessing)
        gt_raw = np.load(ATL / "dataset" / "data" / "ShapeNetV1PointCloud"
                         / cat / f"{obj}.npy").astype(np.float64)[:, :3]
        gt = unit_ball(gt_raw)
        t0 = time.time()
        with torch.no_grad():
            scene_codes = model([pil], device=device)
            mesh = model.extract_mesh(scene_codes, has_vertex_color=False,
                                      resolution=args.mc_resolution)[0]
        pts = np.asarray(trimesh.sample.sample_surface(mesh, args.points)[0],
                         dtype=np.float64)
        pred = unit_ball(pts)
        chamfer, fscore = chamfer_and_fscore(pred, gt, tau_sq=TAU_SQ)
        dt = time.time() - t0
        rows.append({
            "object": f"{cat}_{obj}", "category": cat, "view": args.view,
            "preprocessing": args.preprocessing, "points": args.points,
            "mc_resolution": args.mc_resolution, "seed": 0,
            "chamfer": chamfer, "fscore": round(fscore, 6),
            "seconds": round(dt, 1),
        })
        td = trip / f"{cat}_{obj}"
        td.mkdir(parents=True, exist_ok=True)
        np.save(td / "pred.npy", np.asarray(pred, dtype=np.float32))
        np.save(td / "gt.npy", np.asarray(gt, dtype=np.float32))
        pil.save(td / "input.png")
        print(f"  {cat:16s} {obj:22s} chamfer={chamfer:.4f} "
              f"fscore={fscore:.4f} ({dt:.0f}s)", flush=True)

    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    cat_rows = [(c, float(np.mean([r["chamfer"] for r in rs])),
                  float(np.mean([r["fscore"] for r in rs])),
                  len({r["object"] for r in rs}))
                 for c, rs in sorted(cats.items())]
    noise = sorted(c for c, _, _, n in cat_rows if n < 3)
    try:
        snap = Path(snapshot_download(repo_id="stabilityai/TripoSR",
                                      allow_patterns=["*.ckpt", "*.yaml"]))
        cache_files = []
        for pt in sorted(snap.rglob("*.ckpt")) + sorted(snap.rglob("*.yaml")):
            cache_files.append({"file": pt.name, "bytes": pt.stat().st_size,
                                "sha256": sha256_file(pt)})
    except Exception as exc:
        cache_files = [{"error": f"cache introspection failed: {exc}"}]
    summary = {
        "method": "TripoSR feed-forward (triplane-NeRF + marching cubes)",
        "config": {"view": args.view, "preprocessing": args.preprocessing,
                   "points": args.points, "mc_resolution": args.mc_resolution,
                   "tau_squared": TAU_SQ,
                   "tau_euclid_approx": round(TAU_EUCLID_APPROX, 4),
                   "normalization": "subtract mean, divide by max radius (unit sphere)"},
        "weights": {"repo": "stabilityai/TripoSR", "backbone": "frozen",
                    "fallback": "none", "cache_files": cache_files,
                    "marching_cubes": mc_backend,
                    "src_commit": "107cefdc244c39106fa830359024f6a2f1c78871"},
        "n_rows": len(rows), "n_objects": len({r["object"] for r in rows}),
        "micro_chamfer": float(np.mean([r["chamfer"] for r in rows])),
        "micro_fscore": float(np.mean([r["fscore"] for r in rows])),
        "macro_chamfer": float(np.mean([m[1] for m in cat_rows])) if cat_rows else None,
        "macro_fscore": float(np.mean([m[2] for m in cat_rows])) if cat_rows else None,
        "noise_categories_lt3": noise,
        "per_category": [{"category": c, "n": n, "chamfer": ch, "fscore": fs}
                         for c, ch, fs, n in cat_rows],
    }
    suffix = f"_{args.tag}" if args.tag else ""
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
