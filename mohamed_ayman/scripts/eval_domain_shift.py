#!/usr/bin/env python3
"""
Evaluates Augmented vs Non-Augmented Pix2Vox++ models on non-white cluttered/colored backgrounds
(simulated real-world domain shift) across the 900 unseen test objects.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

TRACK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRACK_ROOT))

from src.data.splits import create_or_load_splits, HOME_DECOR_CATEGORIES
from src.data.voxelizer import normalize_point_cloud, points_to_voxels
from src.losses.voxel_losses import voxel_iou
from src.models.loader import load_pix2vox_checkpoint


def load_domain_shifted_views(obj_dir: Path, seed_rng: random.Random, view_indices=(0, 8, 16)) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    # Consistent room-like background color gradient per object
    c1 = np.array([seed_rng.randint(70, 210) for _ in range(3)], dtype=np.float32)
    c2 = np.array([seed_rng.randint(70, 210) for _ in range(3)], dtype=np.float32)
    t_row = np.linspace(0.0, 1.0, 224, dtype=np.float32)[:, None, None]
    grad = np.broadcast_to((1.0 - t_row) * c1 + t_row * c2, (224, 224, 3)).copy()
    bg_arr = np.concatenate([grad.astype(np.uint8), np.full((224, 224, 1), 255, dtype=np.uint8)], axis=-1)
    bg = Image.fromarray(bg_arr, mode="RGBA")

    for v in view_indices:
        p = obj_dir / f"{v:03d}.png"
        if not p.is_file():
            p = obj_dir / "000.png"
        with Image.open(p) as im:
            im = im.convert("RGBA").resize((224, 224), Image.Resampling.BILINEAR)
            comp = Image.alpha_composite(bg, im).convert("RGB")
            arr = (np.array(comp, dtype=np.float32) / 255.0 - mean) / std
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-ckpt", type=Path, required=True)
    ap.add_argument("--noaug-ckpt", type=Path, required=True)
    ap.add_argument("--max-objects", type=int, default=300)
    ap.add_argument("--out-json", type=Path, default=TRACK_ROOT / "outputs/evaluations/augmentation_ablation_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_aug = load_pix2vox_checkpoint(str(args.aug_ckpt), device=device, use_refiner=True)
    m_noaug = load_pix2vox_checkpoint(str(args.noaug_ckpt), device=device, use_refiner=True)

    splits = create_or_load_splits(dataset_root=str(TRACK_ROOT.parent / "dataset"))
    samples = splits["test"][: args.max_objects]
    rng = random.Random(2026)

    clean_aug_ious, clean_noaug_ious = [], []
    shift_aug_ious, shift_noaug_ious = [], []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    white_bg = Image.new("RGBA", (224, 224), (255, 255, 255, 255))

    for s in tqdm(samples, desc="Domain-Shift Eval"):
        cat, oid = s["category"], s["object_id"]
        obj_dir = TRACK_ROOT.parent / "dataset" / "renders" / cat / oid
        pc_path = TRACK_ROOT.parent / "dataset" / "point_clouds" / cat / f"{oid}.npy"
        cache_v = TRACK_ROOT.parent / "dataset" / "cache" / "voxels_32" / f"{cat}_{oid}.npy"

        if cache_v.is_file():
            vox_gt = np.load(cache_v)
        else:
            pc_n, _, _ = normalize_point_cloud(np.load(pc_path))
            vox_gt = points_to_voxels(pc_n, voxel_res=32, dilate=True)
        gt_t = torch.from_numpy(vox_gt).unsqueeze(0).to(device)

        # Clean white background
        c_tensors = []
        for v in (0, 8, 16):
            with Image.open(obj_dir / f"{v:03d}.png") as im:
                comp = Image.alpha_composite(white_bg, im.convert("RGBA").resize((224, 224), Image.Resampling.BILINEAR)).convert("RGB")
                arr = (np.array(comp, dtype=np.float32) / 255.0 - mean) / std
                c_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
        x_clean = torch.stack(c_tensors, dim=0).unsqueeze(0).to(device)
        x_shift = load_domain_shifted_views(obj_dir, rng).unsqueeze(0).to(device)

        with torch.no_grad():
            clean_aug_ious.append(float(voxel_iou(m_aug(x_clean)["voxels"], gt_t, 0.4)[0].item()))
            clean_noaug_ious.append(float(voxel_iou(m_noaug(x_clean)["voxels"], gt_t, 0.4)[0].item()))
            shift_aug_ious.append(float(voxel_iou(m_aug(x_shift)["voxels"], gt_t, 0.4)[0].item()))
            shift_noaug_ious.append(float(voxel_iou(m_noaug(x_shift)["voxels"], gt_t, 0.4)[0].item()))

    res = {
        "n_test_objects": len(samples),
        "clean_white_bg": {
            "pix2voxpp_noaug_iou": float(np.mean(clean_noaug_ious)),
            "pix2voxpp_aug_iou": float(np.mean(clean_aug_ious)),
        },
        "domain_shift_nonwhite_bg": {
            "pix2voxpp_noaug_iou": float(np.mean(shift_noaug_ious)),
            "pix2voxpp_aug_iou": float(np.mean(shift_aug_ious)),
        },
    }
    args.out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
