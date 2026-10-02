"""
AtlasNet Architecture for Multi-View 3D Surface Reconstruction.
Reference: Groueix et al., "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation", CVPR 2018.

Pipeline:
1. Multi-View 2D CNN Encoder (ResNet-50) extracts view features, aggregated via max-pooling into global latent code z.
2. Ensemble of P=25 MLP patch deformers maps 2D square coordinates (u, v) into 3D Euclidean space.
3. Generates both dense 3D point clouds (4096 points) and continuous triangular surface meshes with fixed patch topology.
"""

import os
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "torch"))

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class PatchDeformerMLP(nn.Module):
    """
    MLP that deforms a 2D planar patch point (u, v) conditioned on global shape code z into 3D coordinates (x, y, z).
    """
    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()  # Output in [-1, 1], then scaled to [-0.5, 0.5]
        )

    def forward(self, points_2d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points_2d: (B, M, 2) 2D coordinates in [0, 1]
            z: (B, latent_dim) global latent code
        Returns:
            points_3d: (B, M, 3) 3D coordinates
        """
        B, M, _ = points_2d.shape
        latent_dim = z.shape[1]

        # Expand z to match point count M: (B, M, latent_dim)
        z_expanded = z.unsqueeze(1).expand(-1, M, -1)

        # Concatenate: (B, M, 2 + latent_dim)
        inp = torch.cat([points_2d, z_expanded], dim=2)

        # Flatten B and M for BatchNorm1d: (B*M, 2 + latent_dim)
        inp_flat = inp.view(B * M, 2 + latent_dim)
        out_flat = self.net(inp_flat)

        # Reshape to (B, M, 3) and scale to [-0.5, 0.5]
        out = (out_flat.view(B, M, 3)) * 0.5
        return out


class AtlasNet(nn.Module):
    """
    Complete Multi-View AtlasNet model with 25 surface patches.
    """
    def __init__(
        self,
        num_patches: int = 25,
        latent_dim: int = 1024,
        hidden_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim = latent_dim

        # Backbone: ResNet-50
        base = None
        if pretrained:
            try:
                base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            except Exception as e:
                print(f"Notice: Could not download online ResNet-50 weights ({e}). Initializing from scratch.")
                base = models.resnet50(weights=None)
        else:
            base = models.resnet50(weights=None)
        self.encoder_conv = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool
        )
        self.fc_latent = nn.Linear(2048, latent_dim)

        # P MLP patch deformers
        self.deformers = nn.ModuleList([
            PatchDeformerMLP(latent_dim=latent_dim, hidden_dim=hidden_dim)
            for _ in range(num_patches)
        ])

    def encode_multiview(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-view features and pools across views to get a single global latent code z.
        
        Args:
            images: (B, K, 3, 224, 224)
        Returns:
            z: (B, latent_dim)
        """
        B, K, C, H, W = images.shape
        x_flat = images.view(B * K, C, H, W)
        feats = self.encoder_conv(x_flat)  # (B*K, 2048, 1, 1)
        feats = torch.flatten(feats, 1)    # (B*K, 2048)
        latents = self.fc_latent(feats)    # (B*K, latent_dim)

        latents_k = latents.view(B, K, self.latent_dim)
        # Multi-view max pooling across views
        z, _ = torch.max(latents_k, dim=1)  # (B, latent_dim)
        return z

    def sample_patch_points(
        self,
        points_per_patch: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Samples 2D uniform random coordinates in [0, 1]^2 for each patch.
        Returns: (num_patches, points_per_patch, 2)
        """
        return torch.rand((self.num_patches, points_per_patch, 2), device=device)

    def forward(
        self,
        images: torch.Tensor,
        num_points: int = 4096
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, K, 3, 224, 224)
            num_points: total points to generate (e.g. 4096)
        Returns:
            dict containing 'points': (B, num_points, 3)
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode multi-view images to global code z
        z = self.encode_multiview(images)  # (B, latent_dim)

        # 2. Sample 2D points on the P patches
        pts_per_patch = int(np.ceil(num_points / self.num_patches))
        patch_pts_2d = self.sample_patch_points(pts_per_patch, device)  # (P, M, 2)

        # 3. Deform each patch into 3D
        deformed_patches = []
        for p in range(self.num_patches):
            pts_2d = patch_pts_2d[p].unsqueeze(0).expand(B, -1, -1)  # (B, M, 2)
            pts_3d = self.deformers[p](pts_2d, z)                   # (B, M, 3)
            deformed_patches.append(pts_3d)

        # Concatenate all patches: (B, P*M, 3)
        all_points = torch.cat(deformed_patches, dim=1)
        # Slice to exact num_points
        out_points = all_points[:, :num_points, :]

        return {
            "points": out_points,
            "latent": z
        }

    def generate_mesh(
        self,
        images: torch.Tensor,
        grid_res: int = 13
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates continuous triangular surface mesh for the first item in the batch.
        
        Args:
            images: (1, K, 3, 224, 224)
            grid_res: resolution of regular grid per patch (13x13 = 169 verts per patch)
        Returns:
            vertices: (N_verts, 3) numpy array
            faces: (N_faces, 3) numpy array of triangle indices
        """
        self.eval()
        device = images.device
        with torch.no_grad():
            z = self.encode_multiview(images)  # (1, latent_dim)

            # Create standard 2D regular grid on [0, 1]^2
            u = torch.linspace(0, 1, grid_res, device=device)
            v = torch.linspace(0, 1, grid_res, device=device)
            grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
            grid_2d = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1).unsqueeze(0)  # (1, grid_res^2, 2)

            # Standard triangulation for grid
            patch_faces = []
            for i in range(grid_res - 1):
                for j in range(grid_res - 1):
                    v0 = i * grid_res + j
                    v1 = v0 + 1
                    v2 = (i + 1) * grid_res + j
                    v3 = v2 + 1
                    patch_faces.append([v0, v2, v1])
                    patch_faces.append([v1, v2, v3])
            patch_faces = np.array(patch_faces, dtype=np.int32)

            all_verts = []
            all_faces = []
            vert_offset = 0

            for p in range(self.num_patches):
                deformed = self.deformers[p](grid_2d, z)  # (1, M, 3)
                v_np = deformed[0].cpu().numpy()
                all_verts.append(v_np)
                all_faces.append(patch_faces + vert_offset)
                vert_offset += v_np.shape[0]

            mesh_verts = np.concatenate(all_verts, axis=0)
            mesh_faces = np.concatenate(all_faces, axis=0)
            return mesh_verts, mesh_faces
