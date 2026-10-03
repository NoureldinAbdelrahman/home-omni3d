"""
Multi-Model Evaluation & Benchmarking Script.
Evaluates Pix2Vox, AtlasNet, and Modern Baselines on the test split.
Outputs cross-representation metrics (Voxel IoU, Chamfer Distance, F-Score@1%, F-Score@2%)
with Macro and Micro class averaging.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.splits import create_or_load_splits, HOME_DECOR_CATEGORIES
from src.data.dataset import OmniObject3DDataset
from src.data.voxelizer import points_to_voxels
from src.models.pix2vox import Pix2Vox
from src.models.atlasnet import AtlasNet
from src.models.baseline_point_e import PointEBaseline
from src.metrics.evaluator import ReconstructionEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D Reconstruction Models")
    parser.add_argument("--model", type=str, default="pix2vox++", choices=["pix2vox", "pix2vox++", "pix2vox_plus", "atlasnet", "point_e"])
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluations")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset directory")
    return parser.parse_args()


def voxels_to_point_cloud(voxels: torch.Tensor, threshold: float = 0.4, num_points: int = 4096) -> torch.Tensor:
    """
    Samples 3D surface points from occupied voxels for cross-representation Chamfer evaluation.
    Args:
        voxels: (B, 32, 32, 32)
    Returns:
        points: (B, num_points, 3)
    """
    B, V, _, _ = voxels.shape
    device = voxels.device
    batch_pts = []

    for b in range(B):
        idx = torch.nonzero(voxels[b] > threshold, as_tuple=False).float()  # (M, 3)
        if len(idx) < 10:
            # Fallback to random points if empty
            pts = torch.rand((num_points, 3), device=device) - 0.5
        else:
            # Rescale index [0, V-1] to [-0.5, 0.5]
            coords = (idx / (V - 1.0)) - 0.5
            if len(coords) >= num_points:
                perm = torch.randperm(len(coords))[:num_points]
                pts = coords[perm]
            else:
                repeats = int(np.ceil(num_points / len(coords)))
                coords_rep = coords.repeat(repeats, 1)
                pts = coords_rep[:num_points]
        batch_pts.append(pts)

    return torch.stack(batch_pts, dim=0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Running Evaluation: {args.model.upper()} ({args.num_views} views) on {device} ===")

    splits = create_or_load_splits(dataset_root=args.dataset_root)
    test_samples = splits["test"]

    test_ds = OmniObject3DDataset(
        samples=test_samples,
        dataset_root=args.dataset_root,
        num_views=args.num_views,
        is_train=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    evaluator = ReconstructionEvaluator(home_decor_categories=HOME_DECOR_CATEGORIES)

    if "pix2vox" in args.model:
        use_refiner = ("++" in args.model) or ("plus" in args.model)
        model = Pix2Vox(pretrained=False, use_refiner=use_refiner).to(device)
        print(f"Evaluating {'Pix2Vox++ (with 3D Refiner)' if use_refiner else 'Pix2Vox (Base / No Refiner)'}")
        if args.weights and os.path.exists(args.weights):
            ckpt = torch.load(args.weights, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {args.weights}")
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Pix2Vox"):
                images = batch["images"].to(device)
                gt_voxels = batch["voxels"].to(device)
                gt_pc = batch["point_cloud"].to(device)

                out = model(images)
                pred_voxels = out["voxels"]
                pred_pc = voxels_to_point_cloud(pred_voxels, threshold=0.4, num_points=4096)

                evaluator.add_batch(
                    categories=batch["category"],
                    object_ids=batch["object_id"],
                    pred_voxels=pred_voxels,
                    gt_voxels=gt_voxels,
                    pred_pc=pred_pc,
                    gt_pc=gt_pc
                )

    elif args.model == "atlasnet":
        model = AtlasNet(num_patches=25, latent_dim=1024, hidden_dim=256, pretrained=False).to(device)
        if args.weights and os.path.exists(args.weights):
            ckpt = torch.load(args.weights, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {args.weights}")
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating AtlasNet"):
                images = batch["images"].to(device)
                gt_pc = batch["point_cloud"].to(device)
                gt_voxels = batch["voxels"].to(device)

                out = model(images, num_points=4096)
                pred_pc = out["points"]

                # Cross-representation: convert AtlasNet point cloud to voxel grid for IoU
                pred_voxels_list = []
                for b in range(pred_pc.shape[0]):
                    pc_np = pred_pc[b].cpu().numpy()
                    v_np = points_to_voxels(pc_np, voxel_res=32, dilate=True)
                    pred_voxels_list.append(torch.from_numpy(v_np))
                pred_voxels = torch.stack(pred_voxels_list, dim=0).to(device)

                evaluator.add_batch(
                    categories=batch["category"],
                    object_ids=batch["object_id"],
                    pred_voxels=pred_voxels,
                    gt_voxels=gt_voxels,
                    pred_pc=pred_pc,
                    gt_pc=gt_pc
                )

    elif args.model == "point_e":
        baseline = PointEBaseline(device=str(device))
        baseline.load_model()

        for batch in tqdm(test_loader, desc="Evaluating Point-E Baseline"):
            gt_pc = batch["point_cloud"].to(device)
            gt_voxels = batch["voxels"].to(device)
            cats = batch["category"]
            obj_ids = batch["object_id"]

            pred_pcs = []
            pred_voxs = []
            for b in range(len(cats)):
                # Use first view image
                render_path = f"dataset/renders/{cats[b]}/{obj_ids[b]}/000.png"
                with Image.open(render_path) as im:
                    pc_np = baseline.reconstruct(im, num_points=4096)
                v_np = points_to_voxels(pc_np, voxel_res=32, dilate=True)
                pred_pcs.append(torch.from_numpy(pc_np))
                pred_voxs.append(torch.from_numpy(v_np))

            pred_pc = torch.stack(pred_pcs, dim=0).to(device)
            pred_voxels = torch.stack(pred_voxs, dim=0).to(device)

            evaluator.add_batch(
                categories=cats,
                object_ids=obj_ids,
                pred_voxels=pred_voxels,
                gt_voxels=gt_voxels,
                pred_pc=pred_pc,
                gt_pc=gt_pc
            )

    results = evaluator.summarize()
    csv_file = os.path.join(args.output_dir, f"{args.model}_{args.num_views}views_test_results.csv")
    results["detailed_df"].to_csv(csv_file, index=False)

    md_report = f"# Evaluation Report: {args.model.upper()} ({args.num_views} Views)\n\n"
    md_report += "## Overall Benchmark Summary\n\n"
    md_report += "| Metric | Overall (Micro) | Per-Class (Macro) | Home Decor Subset | Everyday Objects |\n"
    md_report += "| :--- | :--- | :--- | :--- | :--- |\n"

    for m in ["iou@0.4", "iou@0.5", "cd_l2", "cd_l1", "f_score@0.01", "f_score@0.02"]:
        if m in results["micro"]:
            micro_val = results["micro"].get(m, float("nan"))
            macro_val = results["macro"].get(m, float("nan"))
            decor_val = results["home_decor"].get(m, float("nan"))
            other_val = results["everyday_objects"].get(m, float("nan"))
            md_report += f"| **{m}** | {micro_val:.4f} | {macro_val:.4f} | {decor_val:.4f} | {other_val:.4f} |\n"

    report_path = os.path.join(args.output_dir, f"{args.model}_{args.num_views}views_report.md")
    with open(report_path, "w") as f:
        f.write(md_report)

    print(f"\nSaved evaluation results to:\n- {csv_file}\n- {report_path}")
    print("\n" + md_report)


if __name__ == "__main__":
    main()
