#!/usr/bin/env python3
"""Export Pix2Vox (Base) and Pix2Vox++ (Refined) predictions to the team 3D viewer.

    python export_pix2vox_to_viewer.py
    python viewer/view.py                  # view in the browser dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "bones"))
sys.path.insert(0, str(ROOT / "mohamed_ayman"))

from export import ViewerRun, gt_frame, load_gt_raw, normalize_like_gt  # noqa: E402
from gt_align import load_cameras  # noqa: E402
from src.data.voxelizer import normalize_point_cloud  # noqa: E402
from src.models.pix2vox import Pix2Vox  # noqa: E402


def render_to_raw(P: np.ndarray, fit: dict) -> np.ndarray:
    """Inverse of gt_align.apply_alignment."""
    R = np.array(fit["rotation"])
    return ((np.asarray(P, np.float64) - np.array(fit["offset"])) / fit["scale"]) @ R + np.array(fit["center"])


def voxels_to_shared_frame(vox: np.ndarray, obj_id: str, threshold: float = 0.4) -> np.ndarray:
    gt_raw = load_gt_raw(obj_id)
    c, r = gt_frame(gt_raw)
    _, centroid, max_dist = normalize_point_cloud(gt_raw)

    idx = np.argwhere(vox >= threshold).astype(np.float64)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    coords = (idx / 31.0) - 0.5
    raw_pts = (coords / 0.95) * (2.0 * max_dist) + centroid
    pts_shared = (raw_pts - c) / r
    return pts_shared


def load_views(obj_dir: Path, view_indices: list[int] = (0, 8, 16)) -> torch.Tensor | None:
    tensors = []
    for v in view_indices:
        p = obj_dir / f"{v:03d}.png"
        if not p.is_file():
            return None
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, im).convert("RGB")
        comp = comp.resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.array(comp, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0)  # (K, 3, 224, 224)


def parse_args():
    ap = argparse.ArgumentParser(description="Export Pix2Vox models to the team 3D viewer.")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "mohamed_ayman/outputs/checkpoints/pix2vox_3views_home_decor_best.pth",
        help="Path to trained Pix2Vox checkpoint",
    )
    ap.add_argument("--threshold", type=float, default=0.40, help="Voxel binarization threshold (default: 0.40)")
    ap.add_argument(
        "--objects-json",
        type=Path,
        default=ROOT / "results/colmap/sparse_known/objects.json",
        help="Path to objects.json list to export (defaults to colmap evaluation objects)",
    )
    ap.add_argument("--max-objects", type=int, default=0, help="Limit number of objects (0 = all)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading Pix2Vox from {args.checkpoint} onto {device}...")
    model = Pix2Vox(pretrained=False, use_refiner=True).to(device)
    if args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded model checkpoint successfully.")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with uninitialized weights.")
    model.eval()

    # Determine object list
    if args.objects_json.is_file():
        entries = json.loads(args.objects_json.read_text())
        obj_ids = [e["object"] for e in entries]
    else:
        # Fallback to scanning dataset/renders
        obj_ids = []
        for cat_dir in sorted((ROOT / "dataset/renders").glob("*")):
            if cat_dir.is_dir():
                for od in sorted(cat_dir.glob("*")):
                    if od.is_dir():
                        obj_ids.append(f"{cat_dir.name}/{od.name}")

    if args.max_objects > 0:
        obj_ids = obj_ids[: args.max_objects]

    align_path = ROOT / "bones/gt_alignment.json"
    align = json.loads(align_path.read_text()) if align_path.is_file() else {}

    run_base = ViewerRun(
        owner="mohamed-ayman",
        model="Pix2Vox",
        name="base",
        label="Pix2Vox (Base)",
        notes="Multi-view coarse voxel grid (3 views, 32^3) before 3D refinement.",
    )
    run_pp = ViewerRun(
        owner="mohamed-ayman",
        model="Pix2VoxPlusPlus",
        name="refined",
        label="Pix2Vox++ (Refined)",
        notes="Multi-view 3D U-Net refined voxels (3 views, 32^3) with boundary smoothing and cavity filling.",
    )

    print(f"Exporting predictions for {len(obj_ids)} objects to team viewer...")
    valid_count = 0

    for obj_id in tqdm(obj_ids, desc="Exporting Pix2Vox"):
        obj_dir = ROOT / "dataset/renders" / obj_id
        gt_path = ROOT / "dataset/point_clouds" / f"{obj_id}.npy"
        if not obj_dir.is_dir() or not gt_path.is_file():
            continue

        imgs = load_views(obj_dir, view_indices=[0, 8, 16])
        if imgs is None:
            continue

        batch_img = imgs.unsqueeze(0).to(device)  # (1, 3, 3, 224, 224)

        with torch.no_grad():
            out = model(batch_img)
            vox_base = out["coarse_voxels"][0].cpu().numpy()
            vox_pp = out["voxels"][0].cpu().numpy()

        pts_base = voxels_to_shared_frame(vox_base, obj_id, threshold=args.threshold)
        pts_pp = voxels_to_shared_frame(vox_pp, obj_id, threshold=args.threshold)

        # Camera poses for the 3 active views out of 24
        cams, cams_used = None, None
        fit = align.get(obj_id)
        if fit and "rotation" in fit and (obj_dir / "transforms.json").is_file():
            try:
                names, c2w, _, _ = load_cameras(obj_dir)
                cams = render_to_raw(c2w[:, :3, 3], fit)
                cams_used = [i in (0, 8, 16) for i in range(len(names))]
            except Exception:
                cams, cams_used = None, None

        info = {"views used": "3/24 (views 00, 08, 16)", "threshold": f"{args.threshold:.2f}"}

        run_base.add(
            obj_id,
            pts_base,
            frame="normalized",
            info=info,
            cameras=cams,
            cameras_used=cams_used,
        )
        run_pp.add(
            obj_id,
            pts_pp,
            frame="normalized",
            info=info,
            cameras=cams,
            cameras_used=cams_used,
        )
        valid_count += 1

    print(f"Processed {valid_count} objects. Saving Viewer runs to viewer/runs/...")
    run_base.save()
    run_pp.save()
    print("✓ Successfully exported Pix2Vox and Pix2Vox++ to team 3D viewer!")
    print("Run:  python viewer/view.py  to inspect the dashboard.")


if __name__ == "__main__":
    main()
