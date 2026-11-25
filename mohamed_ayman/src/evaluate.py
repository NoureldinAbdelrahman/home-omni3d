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
from src.models.loader import load_pix2vox_checkpoint, load_atlasnet_checkpoint
from src.models.baseline_point_e import PointEBaseline
from src.metrics.evaluator import ReconstructionEvaluator
from src.losses.chamfer_distance import chamfer_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D Reconstruction Models")
    parser.add_argument("--model", type=str, default="pix2vox++", choices=["pix2vox", "pix2vox++", "pix2vox_plus", "pix2vox_pp", "atlasnet", "point_e"])
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--num_views", "--num_view", dest="num_views", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluations")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename prefix tag for report/csv")
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
            pts = torch.rand((num_points, 3), device=device) - 0.5
        else:
            coords = (idx / (V - 1.0)) - 0.5
            if len(coords) >= num_points:
                perm = torch.randperm(len(coords), device=device)[:num_points]
                pts = coords[perm]
            else:
                repeats = int(np.ceil(num_points / len(coords)))
                coords_rep = coords.repeat(repeats, 1)
                pts = coords_rep[:num_points]
        batch_pts.append(pts)

    return torch.stack(batch_pts, dim=0)


def compute_output_diversity(pcs: List[torch.Tensor]) -> float:
    """Computes mean pairwise L1 Chamfer distance across distinct objects' predictions (diagnoses blob collapse)."""
    if not pcs:
        return 0.0
    sample = torch.cat(pcs, dim=0)[:32]
    n = sample.shape[0]
    if n < 4:
        return 0.0
    dists = []
    with torch.no_grad():
        for i in range(0, n - 1, 2):
            _, cd = chamfer_distance(sample[i:i+1], sample[i+1:i+2], distance_type="l1")
            dists.append(float(cd.item()))
    return float(np.mean(dists)) if dists else 0.0


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
        is_train=False,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    evaluator = ReconstructionEvaluator(home_decor_categories=HOME_DECOR_CATEGORIES)
    diversity_pcs: List[torch.Tensor] = []

    if "pix2vox" in args.model:
        use_refiner = ("++" in args.model) or ("plus" in args.model)
        model = load_pix2vox_checkpoint(args.weights, device=device, use_refiner=use_refiner)
        print(f"Evaluating {'Pix2Vox++' if model.use_refiner else 'Pix2Vox (Base)'} arch={model.arch_config} weights={args.weights}")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Pix2Vox"):
                images = batch["images"].to(device)
                gt_voxels = batch["voxels"].to(device)
                gt_pc = batch["point_cloud"].to(device)

                out = model(images)
                pred_voxels = out["voxels"] if model.use_refiner else out["coarse_voxels"]
                pred_pc = voxels_to_point_cloud(pred_voxels, threshold=0.4, num_points=4096)

                if len(diversity_pcs) * args.batch_size < 32:
                    diversity_pcs.append(pred_pc[:, ::4, :].cpu())

                evaluator.add_batch(
                    categories=batch["category"],
                    object_ids=batch["object_id"],
                    pred_voxels=pred_voxels,
                    gt_voxels=gt_voxels,
                    pred_pc=pred_pc,
                    gt_pc=gt_pc
                )

    elif args.model == "atlasnet":
        model = load_atlasnet_checkpoint(args.weights, device=device)
        print(f"Evaluating AtlasNet arch={model.arch_config} weights={args.weights}")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating AtlasNet"):
                images = batch["images"].to(device)
                gt_pc = batch["point_cloud"].to(device)
                gt_voxels = batch["voxels"].to(device)

                out = model(images, num_points=4096)
                pred_pc = out["points"]

                if len(diversity_pcs) * args.batch_size < 32:
                    diversity_pcs.append(pred_pc[:, ::4, :].cpu())

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
    diversity_cd = compute_output_diversity(diversity_pcs)
    file_tag = args.tag or f"{args.model}_{args.num_views}views"
    csv_file = os.path.join(args.output_dir, f"{file_tag}_test_results.csv")
    results["detailed_df"].to_csv(csv_file, index=False)

    n_objs = len(results["detailed_df"])
    n_cats = results["detailed_df"]["category"].nunique() if n_objs else 0
    md_report = f"# Evaluation Report: {args.model.upper()} ({args.num_views} Views)\n\n"
    md_report += f"- **Test objects**: {n_objs} across {n_cats} categories\n"
    md_report += f"- **Inter-object prediction diversity (CD-L1)**: {diversity_cd:.4f}\n\n"
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

    report_path = os.path.join(args.output_dir, f"{file_tag}_report.md")
    with open(report_path, "w") as f:
        f.write(md_report)

    print(f"\nSaved evaluation results to:\n- {csv_file}\n- {report_path}")
    print("\n" + md_report)


if __name__ == "__main__":
    main()
