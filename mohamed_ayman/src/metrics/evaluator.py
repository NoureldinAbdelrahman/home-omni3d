"""
Comprehensive 3D Reconstruction Evaluator.
Computes IoU, Chamfer Distance, F-Score@1%, F-Score@2%, runtime,
and handles long-tail Macro vs Micro averaging across categories.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.losses.chamfer_distance import chamfer_distance, compute_f_score
from src.losses.voxel_losses import voxel_iou


class ReconstructionEvaluator:
    def __init__(self, home_decor_categories: Optional[List[str]] = None):
        self.home_decor_categories = set(home_decor_categories or [])
        self.records: List[Dict[str, Union[str, float]]] = []

    def reset(self):
        self.records = []

    def add_batch(
        self,
        categories: List[str],
        object_ids: List[str],
        pred_voxels: Optional[torch.Tensor] = None,
        gt_voxels: Optional[torch.Tensor] = None,
        pred_pc: Optional[torch.Tensor] = None,
        gt_pc: Optional[torch.Tensor] = None
    ):
        """
        Records batch evaluation metrics.
        """
        B = len(categories)

        # 1. Voxel IoU
        batch_ious_03 = None
        batch_ious_04 = None
        batch_ious_05 = None
        if pred_voxels is not None and gt_voxels is not None:
            _, batch_ious_03 = voxel_iou(pred_voxels, gt_voxels, threshold=0.3)
            _, batch_ious_04 = voxel_iou(pred_voxels, gt_voxels, threshold=0.4)
            _, batch_ious_05 = voxel_iou(pred_voxels, gt_voxels, threshold=0.5)

        # 2. Chamfer Distance & F-Scores
        batch_cd_l2 = None
        batch_cd_l1 = None
        batch_f1_01 = None
        batch_f1_02 = None
        if pred_pc is not None and gt_pc is not None:
            _, batch_cd_l2 = chamfer_distance(pred_pc, gt_pc, distance_type="l2")
            _, batch_cd_l1 = chamfer_distance(pred_pc, gt_pc, distance_type="l1")
            batch_f1_01, _, _ = compute_f_score(pred_pc, gt_pc, threshold=0.01)
            batch_f1_02, _, _ = compute_f_score(pred_pc, gt_pc, threshold=0.02)

        for i in range(B):
            rec = {
                "category": categories[i],
                "object_id": object_ids[i],
                "is_home_decor": categories[i] in self.home_decor_categories
            }
            if batch_ious_04 is not None:
                rec["iou@0.3"] = float(batch_ious_03[i].item())
                rec["iou@0.4"] = float(batch_ious_04[i].item())
                rec["iou@0.5"] = float(batch_ious_05[i].item())
            if batch_cd_l2 is not None:
                rec["cd_l2"] = float(batch_cd_l2[i].item())
                rec["cd_l1"] = float(batch_cd_l1[i].item())
                rec["f_score@0.01"] = float(batch_f1_01[i].item())
                rec["f_score@0.02"] = float(batch_f1_02[i].item())

            self.records.append(rec)

    def summarize(self) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Computes Micro (overall mean), Macro (mean across classes), and Home Decor subsets.
        """
        if not self.records:
            return {}

        df = pd.DataFrame(self.records)
        metric_cols = [c for c in df.columns if c not in ["category", "object_id", "is_home_decor"]]

        # Micro average (overall mean across instances)
        micro_means = df[metric_cols].mean().to_dict()

        # Macro average (per-class mean, then overall mean)
        per_class_df = df.groupby("category")[metric_cols].mean()
        macro_means = per_class_df.mean().to_dict()

        # Home Decor subset mean
        home_decor_df = df[df["is_home_decor"] == True]
        home_decor_means = home_decor_df[metric_cols].mean().to_dict() if len(home_decor_df) > 0 else {}

        # Out-of-domain everyday objects mean
        other_df = df[df["is_home_decor"] == False]
        other_means = other_df[metric_cols].mean().to_dict() if len(other_df) > 0 else {}

        return {
            "micro": micro_means,
            "macro": macro_means,
            "home_decor": home_decor_means,
            "everyday_objects": other_means,
            "per_class_summary": per_class_df,
            "detailed_df": df
        }
