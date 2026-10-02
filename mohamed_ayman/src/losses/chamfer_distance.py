"""
GPU-accelerated Chamfer Distance and F-Score calculation in pure PyTorch.
Compatible with batches of point clouds.
"""

from typing import Tuple
import torch
import torch.nn as nn


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise squared Euclidean distances between points in x and y.
    
    Args:
        x: (B, N, 3)
        y: (B, M, 3)
    Returns:
        dist: (B, N, M)
    """
    # (B, N, 1) + (B, 1, M) - 2 * (B, N, M)
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return torch.clamp(dist, min=0.0)


def chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    distance_type: str = "l2"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates symmetric Chamfer Distance between predicted and ground-truth point clouds.
    
    Args:
        pred: (B, N, 3) predicted point cloud
        gt: (B, M, 3) ground truth point cloud
        distance_type: 'l2' (squared Euclidean) or 'l1' (Euclidean)
    Returns:
        loss: scalar tensor (mean over batch)
        raw_cd: (B,) batch of chamfer values
    """
    sq_dist = pairwise_distances(pred, gt)  # (B, N, M)

    min_pred_to_gt, _ = torch.min(sq_dist, dim=2)  # (B, N)
    min_gt_to_pred, _ = torch.min(sq_dist, dim=1)  # (B, M)

    if distance_type == "l1":
        min_pred_to_gt = torch.sqrt(min_pred_to_gt + 1e-12)
        min_gt_to_pred = torch.sqrt(min_gt_to_pred + 1e-12)

    cd_pred = min_pred_to_gt.mean(dim=1)
    cd_gt = min_gt_to_pred.mean(dim=1)

    batch_cd = cd_pred + cd_gt
    return batch_cd.mean(), batch_cd


def compute_f_score(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Precision, Recall, and F-Score at a given distance threshold (e.g. 1% or 2% of bounding box).
    
    Args:
        pred: (B, N, 3)
        gt: (B, M, 3)
        threshold: distance threshold (default 0.01 for normalized [-0.5, 0.5]^3 scale)
    Returns:
        f_score: (B,) tensor
        precision: (B,) tensor
        recall: (B,) tensor
    """
    with torch.no_grad():
        sq_dist = pairwise_distances(pred, gt)
        dist_pred_to_gt = torch.sqrt(sq_dist.min(dim=2)[0] + 1e-12)  # (B, N)
        dist_gt_to_pred = torch.sqrt(sq_dist.min(dim=1)[0] + 1e-12)  # (B, M)

        precision = (dist_pred_to_gt < threshold).float().mean(dim=1)
        recall = (dist_gt_to_pred < threshold).float().mean(dim=1)

        f_score = 2.0 * precision * recall / (precision + recall + 1e-8)
        return f_score, precision, recall


class ChamferLoss(nn.Module):
    def __init__(self, distance_type: str = "l2"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss, _ = chamfer_distance(pred, gt, distance_type=self.distance_type)
        return loss
