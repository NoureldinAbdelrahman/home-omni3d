"""
Unified Modular Training Pipeline for Multi-View 3D Shape Reconstruction.
Supports Pix2Vox++ (volumetric) and AtlasNet (spherical/planar surface patch)
with PyTorch AMP, 3D-safe image augmentation, class-balanced long-tail sampling,
and multi-view K-dropout for K in {1, 3, 5} generalization.
"""

import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from typing import Dict, Any, Optional

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.splits import create_or_load_splits, HOME_DECOR_CATEGORIES
from src.data.dataset import OmniObject3DDataset
from src.models.pix2vox import Pix2Vox, Pix2VoxPlusPlus
from src.models.atlasnet import AtlasNet
from src.losses.voxel_losses import Pix2VoxLoss, voxel_iou
from src.losses.chamfer_distance import ChamferLoss, chamfer_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-View 3D Reconstruction Models")
    parser.add_argument("--model", type=str, default="pix2vox++", choices=["pix2vox", "pix2vox++", "pix2vox_plus", "pix2vox_pp", "atlasnet"])
    parser.add_argument("--num_views", "--num_view", dest="num_views", type=int, default=3, help="Max RGB input viewpoints (1, 3, 5)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--subset", type=str, default="all", choices=["home_decor", "all"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True, help="Enable 3D-safe multi-view image augmentation")
    parser.add_argument("--no_augment", dest="augment", action="store_false", help="Disable image augmentation (baseline ablation)")
    parser.add_argument("--class_balanced", dest="class_balanced", action="store_true", default=True, help="Inverse-sqrt class-balanced sampling for long tail")
    parser.add_argument("--no_class_balanced", dest="class_balanced", action="store_false")
    parser.add_argument("--multi_k", action="store_true", default=False, help="Randomly sample 1..num_views per batch during training")
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet18", "resnet50"])
    parser.add_argument("--spatial_pool", type=int, default=None)
    parser.add_argument("--feat_dim", type=int, default=None)
    parser.add_argument("--deep_refiner", action="store_true", default=False)
    parser.add_argument("--template", type=str, default="sphere", choices=["square", "sphere"])
    parser.add_argument("--view_pooling", type=str, default="attention", choices=["max", "attention"])
    parser.add_argument("--run_name", type=str, default=None, help="Optional checkpoint filename tag")
    return parser.parse_args()


def build_sampler(samples: list) -> WeightedRandomSampler:
    counts = Counter(s["category"] for s in samples)
    weights = [1.0 / (counts[s["category"]] ** 0.5) for s in samples]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(samples), replacement=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    model_type: str,
    device: torch.device,
    use_amp: bool = True,
    multi_k: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    count = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)  # (B, K, 3, 224, 224)
        if multi_k and images.shape[1] > 1:
            k_choices = [1] + [k for k in (2, 3, 3, 3, 5, 5) if k <= images.shape[1]]
            k_pick = random.choice(k_choices)
            if k_pick < images.shape[1]:
                perm = torch.randperm(images.shape[1], device=device)[:k_pick]
                images = images[:, perm]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            if "pix2vox" in model_type:
                gt_voxels = batch["voxels"].to(device, non_blocking=True)  # (B, 32, 32, 32)
                out = model(images)
                loss = criterion(out["voxels"], gt_voxels, out["coarse_voxels"])
                metric, _ = voxel_iou(out["voxels"], gt_voxels, threshold=0.4)
            else:  # atlasnet
                gt_pc = batch["point_cloud"].to(device, non_blocking=True)  # (B, 4096, 3)
                out = model(images, num_points=4096)
                loss = criterion(out["points"], gt_pc)
                _, batch_cd = chamfer_distance(out["points"], gt_pc, distance_type="l2")
                metric = batch_cd.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        b_size = images.shape[0]
        total_loss += loss.item() * b_size
        total_metric += metric.item() * b_size
        count += b_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "metric": f"{metric.item():.4f}"})

    return {
        "loss": total_loss / max(1, count),
        "metric": total_metric / max(1, count)
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    model_type: str,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            b_size = images.shape[0]

            if "pix2vox" in model_type:
                gt_voxels = batch["voxels"].to(device, non_blocking=True)
                out = model(images)
                loss = criterion(out["voxels"], gt_voxels, out["coarse_voxels"])
                metric, _ = voxel_iou(out["voxels"], gt_voxels, threshold=0.4)
            else:
                gt_pc = batch["point_cloud"].to(device, non_blocking=True)
                out = model(images, num_points=4096)
                loss = criterion(out["points"], gt_pc)
                _, batch_cd = chamfer_distance(out["points"], gt_pc, distance_type="l2")
                metric = batch_cd.mean()

            total_loss += loss.item() * b_size
            total_metric += metric.item() * b_size
            count += b_size

    return {
        "loss": total_loss / max(1, count),
        "metric": total_metric / max(1, count)
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training: {args.model.upper()} ({args.num_views} views, augment={args.augment}, balanced={args.class_balanced}) on {device} ===")

    # 1. Load splits
    splits = create_or_load_splits(dataset_root=args.dataset_root)
    train_samples = splits["train"]
    val_samples = splits["val"]

    if args.subset == "home_decor":
        decor_set = set(HOME_DECOR_CATEGORIES)
        train_samples = [s for s in train_samples if s["category"] in decor_set]
        val_samples = [s for s in val_samples if s["category"] in decor_set]
        print(f"Filtered for Home Decor subset: {len(train_samples)} train, {len(val_samples)} val.")
    else:
        n_cats = len(set(s["category"] for s in train_samples))
        print(f"Full OmniObject3D dataset ({n_cats} categories): {len(train_samples)} train, {len(val_samples)} val.")

    # 2. Datasets & Loaders
    train_ds = OmniObject3DDataset(
        samples=train_samples,
        dataset_root=args.dataset_root,
        num_views=args.num_views,
        is_train=True,
        augment=args.augment,
    )
    val_ds = OmniObject3DDataset(
        samples=val_samples,
        dataset_root=args.dataset_root,
        num_views=args.num_views,
        is_train=False,
        augment=False,
    )

    sampler = build_sampler(train_samples) if args.class_balanced else None
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )

    # 3. Model & Loss setup
    if "pix2vox" in args.model:
        use_refiner = ("++" in args.model) or ("plus" in args.model)
        backbone = args.backbone or ("resnet50" if args.deep_refiner or args.spatial_pool else "resnet18")
        spatial_pool = args.spatial_pool or (2 if backbone == "resnet50" else 1)
        feat_dim = args.feat_dim or (1024 if (backbone == "resnet50" or spatial_pool > 1) else 512)
        model = Pix2Vox(
            pretrained=True,
            use_refiner=use_refiner,
            backbone=backbone,
            spatial_pool=spatial_pool,
            feat_dim=feat_dim,
            deep_refiner=args.deep_refiner,
        ).to(device)
        criterion = Pix2VoxLoss(bce_weight=1.0, dice_weight=0.5).to(device)
        metric_name = "Voxel_IoU"
        best_metric = -1.0  # higher is better
        print(f"Instantiated {'Pix2Vox++' if use_refiner else 'Pix2Vox'} ({model.arch_config})")
    else:
        model = AtlasNet(
            num_patches=25,
            latent_dim=1024,
            hidden_dim=256,
            pretrained=True,
            template=args.template,
            view_pooling=args.view_pooling,
        ).to(device)
        criterion = ChamferLoss(distance_type="l2").to(device)
        metric_name = "Chamfer_L2"
        best_metric = 1e9   # lower is better
        print(f"Instantiated AtlasNet ({model.arch_config})")

    # Parameter groups (lower lr for pretrained encoder, higher for decoder)
    encoder_params = list(model.encoder.parameters()) if hasattr(model, "encoder") else list(model.encoder_conv.parameters())
    decoder_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": decoder_params, "lr": args.lr_decoder}
    ], weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=args.mixed_precision)

    os.makedirs(args.save_dir, exist_ok=True)
    history = {"train_loss": [], "train_metric": [], "val_loss": [], "val_metric": []}
    run_tag = args.run_name or f"{args.model}_{args.num_views}views_{args.subset}"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            args.model, device, use_amp=args.mixed_precision, multi_k=args.multi_k
        )
        val_res = validate(
            model, val_loader, criterion, args.model, device
        )
        lr_scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_res["loss"])
        history["train_metric"].append(train_res["metric"])
        history["val_loss"].append(val_res["loss"])
        history["val_metric"].append(val_res["metric"])

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_res['loss']:.4f}, {metric_name}: {train_res['metric']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, {metric_name}: {val_res['metric']:.4f}"
        )

        is_best = (val_res["metric"] > best_metric) if "pix2vox" in args.model else (val_res["metric"] < best_metric)
        if is_best:
            best_metric = val_res["metric"]
            best_path = os.path.join(args.save_dir, f"{run_tag}_best.pth")
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "model_type": args.model,
                "num_views": args.num_views,
                "arch_config": getattr(model, "arch_config", {}),
                "augment": args.augment,
                "subset": args.subset,
            }
            torch.save(ckpt_dict, best_path)
            print(f"  -> Saved new best model ({metric_name}: {best_metric:.4f}) to {best_path}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, f"{run_tag}_latest.pth"))
    with open(os.path.join(args.save_dir, f"{run_tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"=== Completed Training {run_tag}! Best {metric_name}: {best_metric:.4f} ===")


if __name__ == "__main__":
    main()
