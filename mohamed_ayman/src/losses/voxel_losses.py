"""
Loss functions and evaluation metrics for 3D Voxel Occupancy Grids (Pix2Vox).
Handles volumetric class imbalance via Weighted BCE and Soft Dice Loss.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def voxel_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes volumetric Intersection-over-Union (IoU) between predicted probabilities and ground truth voxels.
    
    Args:
        pred: (B, V, V, V) predicted probabilities in [0, 1]
        gt: (B, V, V, V) binary ground truth in {0, 1}
        threshold: binarization threshold (default 0.4)
    Returns:
        mean_iou: scalar tensor
        batch_iou: (B,) tensor
    """
    with torch.no_grad():
        bin_pred = (pred > threshold).float()
        bin_gt = (gt > 0.5).float()

        intersection = (bin_pred * bin_gt).sum(dim=(1, 2, 3))
        union = ((bin_pred + bin_gt) > 0).float().sum(dim=(1, 2, 3))

        # Avoid 0 / 0 division for completely empty ground truths
        batch_iou = (intersection + 1e-6) / (union + 1e-6)
        return batch_iou.mean(), batch_iou


class SoftDiceLoss(nn.Module):
    """
    Volumetric Soft Dice Loss for handling extreme sparsity in 3D voxel grids.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims: (B, -1)
        pred_flat = pred.contiguous().view(pred.shape[0], -1)
        gt_flat = gt.contiguous().view(gt.shape[0], -1)

        intersection = 2.0 * (pred_flat * gt_flat).sum(dim=1) + self.smooth
        denominator = (pred_flat.pow(2) + gt_flat.pow(2)).sum(dim=1) + self.smooth

        dice = intersection / denominator
        return (1.0 - dice).mean()


class Pix2VoxLoss(nn.Module):
    """
    Combined loss for Pix2Vox: Weighted Binary Cross-Entropy + Soft Dice Loss.
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        pos_weight: float = 2.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice_loss = SoftDiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        coarse_pred: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 32, 32, 32) probabilities from refiner / merger
            gt: (B, 32, 32, 32) target voxels
            coarse_pred: optional (B, 32, 32, 32) probabilities before refiner
        """
        # Weighted BCE
        eps = 1e-7
        p_clamped = torch.clamp(pred, eps, 1.0 - eps)
        bce = - (self.pos_weight * gt * torch.log(p_clamped) + (1.0 - gt) * torch.log(1.0 - p_clamped))
        bce_loss = bce.mean()

        dice_loss = self.dice_loss(pred, gt)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        if coarse_pred is not None:
            # Auxiliary supervision on coarse stage
            c_clamped = torch.clamp(coarse_pred, eps, 1.0 - eps)
            c_bce = - (self.pos_weight * gt * torch.log(c_clamped) + (1.0 - gt) * torch.log(1.0 - c_clamped)).mean()
            c_dice = self.dice_loss(coarse_pred, gt)
            total_loss = total_loss + 0.5 * (self.bce_weight * c_bce + self.dice_weight * c_dice)

        return total_loss
