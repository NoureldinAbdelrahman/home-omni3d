"""
Pix2Vox / Pix2Vox++ Architecture for Multi-View 3D Shape Reconstruction.
Reference: Xie et al., "Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images", ICCV 2019,
and "Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images", IJCV 2020.

Pipeline:
1. Shared 2D CNN Encoder (ResNet-18 or ResNet-50 with optional spatial pooling) extracts feature vectors from K views.
2. 3D Deconvolution Decoder generates coarse 32x32x32 voxel grids and visibility scores.
3. Context-Aware Multi-View Fusion adaptively weights voxel features across viewpoints.
4. 3D U-Net / Residual Refiner refines surface boundaries and fills inner cavities.
"""

import os
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "torch"))

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder2D(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        backbone: str = "resnet18",
        spatial_pool: int = 1,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.spatial_pool = spatial_pool
        self.feat_dim = feat_dim

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            try:
                base = models.resnet50(weights=weights)
            except Exception as e:
                print(f"Notice: Could not download online ResNet-50 weights ({e}). Initializing from scratch.")
                base = models.resnet50(weights=None)
            enc_channels = 2048
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                base = models.resnet18(weights=weights)
            except Exception as e:
                print(f"Notice: Could not download online ResNet-18 weights ({e}). Initializing from scratch.")
                base = models.resnet18(weights=None)
            enc_channels = 512

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((spatial_pool, spatial_pool))

        in_flat = enc_channels * (spatial_pool * spatial_pool)
        if spatial_pool == 1 and backbone == "resnet18" and feat_dim == 512:
            # Keep exact attribute name for legacy checkpoint compatibility
            self.fc = nn.Linear(in_flat, feat_dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_flat, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, 3, 224, 224)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.fc(x)  # (B*K, feat_dim)
        return feat


class Decoder3D(nn.Module):
    def __init__(self, in_features: int = 512):
        super().__init__()
        # Project in_features -> 256 * 2^3 = 2048
        self.fc = nn.Linear(in_features, 256 * 2 * 2 * 2)

        # Transposed 3D Convolutions: 2 -> 4 -> 8 -> 16 -> 32
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # Outputs 2 channels: channel 0 = raw voxel logit, channel 1 = view confidence score
        self.deconv5 = nn.Conv3d(16, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B*K, in_features)
        x = self.fc(feat).view(-1, 256, 2, 2, 2)
        x = self.deconv1(x)  # (B*K, 128, 4, 4, 4)
        x = self.deconv2(x)  # (B*K, 64, 8, 8, 8)
        x = self.deconv3(x)  # (B*K, 32, 16, 16, 16)
        x = self.deconv4(x)  # (B*K, 16, 32, 32, 32)
        out = self.deconv5(x)  # (B*K, 2, 32, 32, 32)

        raw_voxels = out[:, 0:1, :, :, :]  # (B*K, 1, 32, 32, 32)
        raw_scores = out[:, 1:2, :, :, :]  # (B*K, 1, 32, 32, 32)
        return raw_voxels, raw_scores


class ContextAwareFusion(nn.Module):
    """
    Fuses K view-specific 32x32x32 voxel grids using learned spatial attention weights.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_voxels: torch.Tensor,
        raw_scores: torch.Tensor,
        batch_size: int,
        num_views: int
    ) -> torch.Tensor:
        """
        raw_voxels: (B*K, 1, 32, 32, 32)
        raw_scores: (B*K, 1, 32, 32, 32)
        """
        if num_views == 1:
            return torch.sigmoid(raw_voxels.view(batch_size, 1, 32, 32, 32))

        # Reshape to (B, K, 1, 32, 32, 32)
        voxels_k = raw_voxels.view(batch_size, num_views, 1, 32, 32, 32)
        scores_k = raw_scores.view(batch_size, num_views, 1, 32, 32, 32)

        # Softmax over view dimension K
        weights = F.softmax(scores_k, dim=1)

        # Weighted sum: (B, 1, 32, 32, 32)
        fused = (weights * torch.sigmoid(voxels_k)).sum(dim=1)
        return fused


class Refiner3D(nn.Module):
    """
    3D U-Net / Residual 3D CNN refiner to clean noise and fill hollow cavities.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 16, 16, 16)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # (B, 16, 32, 32, 32)
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 32, 32)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        d = self.deconv(c3)
        res = self.out_conv(d + c1)
        # Residual connection
        refined = torch.sigmoid(x + res)
        return refined


class MultiScaleRefiner3D(nn.Module):
    """
    Deeper 3D U-Net Refiner (32^3 -> 16^3 -> 8^3 -> 16^3 -> 32^3) with logit residual.
    Ensures exact identity mapping at delta=0 and stronger multi-scale cavity completion.
    """
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(b) + e2
        d1 = self.dec1(d2) + e1
        delta = self.out_conv(d1)
        # Perform logit residual in FP32 to prevent float16 clamp rounding to inf/nan
        base_logits = torch.logit(x.float().clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(base_logits + delta.float())


class Pix2Vox(nn.Module):
    """
    Complete Multi-View Pix2Vox / Pix2Vox++ model.
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_refiner: bool = True,
        backbone: str = "resnet18",
        spatial_pool: int = 1,
        feat_dim: int = 512,
        deep_refiner: bool = False,
    ):
        super().__init__()
        self.use_refiner = use_refiner
        self.arch_config = {
            "backbone": backbone,
            "spatial_pool": spatial_pool,
            "feat_dim": feat_dim,
            "deep_refiner": deep_refiner,
            "use_refiner": use_refiner,
        }
        self.encoder = Encoder2D(
            pretrained=pretrained,
            backbone=backbone,
            spatial_pool=spatial_pool,
            feat_dim=feat_dim,
        )
        self.decoder = Decoder3D(in_features=feat_dim)
        self.fusion = ContextAwareFusion()
        if self.use_refiner:
            self.refiner = MultiScaleRefiner3D() if deep_refiner else Refiner3D()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, K, 3, 224, 224)
        Returns:
            dict containing:
                'voxels': (B, 32, 32, 32)
                'coarse_voxels': (B, 32, 32, 32)
        """
        B, K, C, H, W = images.shape
        x_flat = images.view(B * K, C, H, W)
        feats = self.encoder(x_flat)  # (B*K, feat_dim)

        raw_voxels, raw_scores = self.decoder(feats)  # (B*K, 1, 32, 32, 32)
        coarse_fused = self.fusion(raw_voxels, raw_scores, B, K)  # (B, 1, 32, 32, 32)

        if self.use_refiner:
            refined = self.refiner(coarse_fused)  # (B, 1, 32, 32, 32)
            out_voxels = refined.squeeze(1)
        else:
            out_voxels = coarse_fused.squeeze(1)

        return {
            "voxels": out_voxels,
            "coarse_voxels": coarse_fused.squeeze(1)
        }


class Pix2VoxPlusPlus(Pix2Vox):
    """
    Pix2Vox++ Architecture (Xie et al., IJCV 2020).
    Features Context-Aware Multi-View Fusion with Multi-Scale 3D U-Net Refiner.
    """
    def __init__(
        self,
        pretrained: bool = True,
        backbone: str = "resnet50",
        spatial_pool: int = 2,
        feat_dim: int = 1024,
        deep_refiner: bool = True,
    ):
        super().__init__(
            pretrained=pretrained,
            use_refiner=True,
            backbone=backbone,
            spatial_pool=spatial_pool,
            feat_dim=feat_dim,
            deep_refiner=deep_refiner,
        )
