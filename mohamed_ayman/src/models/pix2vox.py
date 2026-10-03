"""
Pix2Vox / Pix2Vox++ Architecture for Multi-View 3D Shape Reconstruction.
Reference: Xie et al., "Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images", ICCV 2019.

Pipeline:
1. Shared 2D CNN Encoder (ResNet-18) extracts feature vectors from K views.
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
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = None
        if pretrained:
            try:
                base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as e:
                print(f"Notice: Could not download online ResNet-18 weights ({e}). Initializing from scratch.")
                base = models.resnet18(weights=None)
        else:
            base = models.resnet18(weights=None)
        # Remove avgpool and fc
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)

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
        feat = self.fc(x)  # (B*K, 512)
        return feat


class Decoder3D(nn.Module):
    def __init__(self, in_features: int = 512):
        super().__init__()
        # Project 512 -> 256 * 2^3 = 2048
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
        # feat: (B*K, 512)
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


class Pix2Vox(nn.Module):
    """
    Complete Multi-View Pix2Vox++ model.
    """
    def __init__(self, pretrained: bool = True, use_refiner: bool = True):
        super().__init__()
        self.use_refiner = use_refiner
        self.encoder = Encoder2D(pretrained=pretrained)
        self.decoder = Decoder3D(in_features=512)
        self.fusion = ContextAwareFusion()
        if self.use_refiner:
            self.refiner = Refiner3D()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, K, 3, 224, 224)
        Returns:
            dict containing:
                'coarse_voxels': (B, 32, 32, 32)
                'refined_voxels': (B, 32, 32, 32)
        """
        B, K, C, H, W = images.shape
        # Flatten B and K for shared 2D encoder
        x_flat = images.view(B * K, C, H, W)
        feats = self.encoder(x_flat)  # (B*K, 512)

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
    Features Context-Aware Multi-View Fusion with 3D Residual U-Net Refiner.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(pretrained=pretrained, use_refiner=True)
