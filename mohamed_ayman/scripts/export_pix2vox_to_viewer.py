#!/usr/bin/env python3
"""Export Pix2Vox++ and AtlasNet predictions to the team 3D viewer.

    python export_pix2vox_to_viewer.py --model pix2vox++ --checkpoint ... --run-name ms3_omni215_aug
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
from src.models.loader import load_pix2vox_checkpoint, load_atlasnet_checkpoint  # noqa: E402


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


def norm_pc_to_shared_frame(norm_pc: np.ndarray, obj_id: str) -> np.ndarray:
    """Maps [-0.5, 0.5]^3 normalized point cloud back to shared unit-sphere GT frame."""
    gt_raw = load_gt_raw(obj_id)
    c, r = gt_frame(gt_raw)
    _, centroid, max_dist = normalize_point_cloud(gt_raw)
    raw_pts = (norm_pc.astype(np.float64) / 0.95) * (2.0 * max_dist) + centroid
    return (raw_pts - c) / r


def load_views(obj_dir: Path, view_indices: list[int] = (0, 8, 16)) -> torch.Tensor | None:
    tensors = []
    for v in view_indices:
        p = obj_dir / f"{v:03d}.png"
        if not p.is_file():
            return None
        im = Image.open(p).convert("RGBA")
        im = im.resize((224, 224), Image.Resampling.BILINEAR)
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, im).convert("RGB")
        arr = (np.array(comp, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0)


def parse_args():
    ap = argparse.ArgumentParser(description="Export Pix2Vox++ / AtlasNet models to the team 3D viewer.")
    ap.add_argument("--model", type=str, default="pix2vox++", choices=["pix2vox++", "atlasnet"])
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "mohamed_ayman/outputs/checkpoints/pix2vox_3views_home_decor_best.pth",
        help="Path to trained model checkpoint",
    )
    ap.add_argument("--run-name", type=str, default="refined", help="Run suffix under viewer/runs/")
    ap.add_argument("--label", type=str, default=None, help="Display label in viewer dropdown")
    ap.add_argument("--notes", type=str, default=None, help="Run description notes")
    ap.add_argument("--threshold", type=float, default=0.40, help="Voxel binarization threshold (default: 0.40)")
    ap.add_argument("--num-views", type=int, default=3, help="Number of views (1, 3, or 5)")
    ap.add_argument(
        "--objects-json",
        type=Path,
        default=ROOT / "results/colmap/sparse_known/objects.json",
        help="Path to objects.json list to export (defaults to colmap evaluation objects)",
    )
    ap.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Optional splits.json path to export test-split objects instead of objects-json",
    )
    ap.add_argument("--max-objects", type=int, default=0, help="Limit number of objects (0 = all)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model} from {args.checkpoint} onto {device}...")
    if args.model == "atlasnet":
        model = load_atlasnet_checkpoint(str(args.checkpoint), device=device)
        run = ViewerRun(
            owner="mohamed-ayman",
            model="AtlasNet",
            name=args.run_name,
            label=args.label or f"AtlasNet ({args.run_name})",
            notes=args.notes or f"Multi-view Surface AtlasNet ({args.num_views} views, arch={model.arch_config}).",
        )
    else:
        model = load_pix2vox_checkpoint(str(args.checkpoint), device=device, use_refiner=True)
        run = ViewerRun(
            owner="mohamed-ayman",
            model="Pix2VoxPlusPlus",
            name=args.run_name,
            label=args.label or f"Pix2Vox++ ({args.run_name})",
            notes=args.notes or f"Multi-view 3D U-Net refined voxels ({args.num_views} views, arch={model.arch_config}).",
        )

    if args.splits_json and args.splits_json.is_file():
        splits = json.loads(args.splits_json.read_text())
        obj_ids = [f"{s['category']}/{s['object_id']}" for s in splits["test"]]
    elif args.objects_json.is_file():
        entries = json.loads(args.objects_json.read_text())
        obj_ids = [e["object"] for e in entries]
    else:
        obj_ids = []
        for cat_dir in sorted((ROOT / "dataset/renders").glob("*")):
            if cat_dir.is_dir():
                for od in sorted(cat_dir.glob("*")):
                    if od.is_dir():
                        obj_ids.append(f"{cat_dir.name}/{od.name}")

    if args.max_objects > 0:
        obj_ids = obj_ids[: args.max_objects]

    step = 24 // max(1, args.num_views)
    view_indices = [(i * step) % 24 for i in range(args.num_views)]

    align_path = ROOT / "bones/gt_alignment.json"
    align = json.loads(align_path.read_text()) if align_path.is_file() else {}

    print(f"Exporting {args.model} predictions for {len(obj_ids)} objects to team viewer...")
    valid_count = 0

    for obj_id in tqdm(obj_ids, desc=f"Exporting {args.model}"):
        obj_dir = ROOT / "dataset/renders" / obj_id
        gt_path = ROOT / "dataset/point_clouds" / f"{obj_id}.npy"
        if not obj_dir.is_dir() or not gt_path.is_file():
            continue

        imgs = load_views(obj_dir, view_indices=view_indices)
        if imgs is None:
            continue

        batch_img = imgs.unsqueeze(0).to(device)
        with torch.no_grad():
            if args.model == "atlasnet":
                out = model(batch_img, num_points=4096)
                norm_pc = out["points"][0].cpu().numpy()
                pts = norm_pc_to_shared_frame(norm_pc, obj_id)
            else:
                out = model(batch_img)
                vox_pp = out["voxels"][0].cpu().numpy()
                pts = voxels_to_shared_frame(vox_pp, obj_id, threshold=args.threshold)

        cams, cams_used = None, None
        fit = align.get(obj_id)
        if fit and "rotation" in fit and (obj_dir / "transforms.json").is_file():
            try:
                names, c2w, _, _ = load_cameras(obj_dir)
                cams = render_to_raw(c2w[:, :3, 3], fit)
                cams_used = [i in view_indices for i in range(len(names))]
            except Exception:
                cams, cams_used = None, None

        info = {
            "views used": f"{args.num_views}/24 (views {', '.join(f'{v:02d}' for v in view_indices)})",
            "model": args.model,
        }
        if args.model != "atlasnet":
            info["threshold"] = f"{args.threshold:.2f}"

        run.add(
            obj_id,
            pts,
            frame="normalized",
            info=info,
            cameras=cams,
            cameras_used=cams_used,
        )
        valid_count += 1

    print(f"Processed {valid_count} objects. Saving Viewer run to viewer/runs/...")
    run.save()
    print(f"✓ Successfully exported {args.model} ({args.run_name}) to team 3D viewer!")


if __name__ == "__main__":
    main()
