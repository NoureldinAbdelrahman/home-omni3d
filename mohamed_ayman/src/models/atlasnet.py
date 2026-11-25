"""
AtlasNet Architecture for Multi-View 3D Surface Reconstruction.
Reference: Groueix et al., "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation", CVPR 2018.

Pipeline:
1. Multi-View 2D CNN Encoder (ResNet-50) extracts view features, aggregated via learned attention pooling into global latent code z.
2. Ensemble of P=25 MLP patch deformers maps either closed spherical coordinates (template="sphere") or 2D planar patches (template="square") into 3D space.
3. Generates both dense 3D point clouds (4096 points) and continuous triangular surface meshes.
"""

import os
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "torch"))

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchDeformerMLP(nn.Module):
    """
    MLP that deforms a template coordinate (2D (u,v) or 3D unit sphere point) conditioned on global shape code z into 3D coordinates (x, y, z).
    """
    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 256, coord_dim: int = 2):
        super().__init__()
        self.coord_dim = coord_dim
        self.net = nn.Sequential(
            nn.Linear(coord_dim + latent_dim, hidden_dim),
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

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, M, coord_dim)
            z: (B, latent_dim) global latent code
        Returns:
            points_3d: (B, M, 3) 3D coordinates in [-0.5, 0.5]
        """
        B, M, _ = coords.shape
        latent_dim = z.shape[1]

        z_expanded = z.unsqueeze(1).expand(-1, M, -1)
        inp = torch.cat([coords, z_expanded], dim=2)
        inp_flat = inp.view(B * M, self.coord_dim + latent_dim)
        out_flat = self.net(inp_flat)
        return out_flat.view(B, M, 3) * 0.5


def _uv_to_sphere_patch(uv: torch.Tensor, patch_idx: int, num_patches: int) -> torch.Tensor:
    """
    Maps local patch UV in [0, 1]^2 onto a clean spherical tile on the unit sphere S^2.
    Divides sphere azimuth x polar angles cleanly across num_patches.
    """
    n_lat = max(1, int(round(np.sqrt(num_patches))))
    n_lon = max(1, int(np.ceil(num_patches / n_lat)))
    row = patch_idx // n_lon
    col = patch_idx % n_lon

    u = uv[..., 0]
    v = uv[..., 1]

    phi = ((col + u) / n_lon) * (2.0 * np.pi)
    # Cosine-weighted latitude bands prevent pole bunching
    c_min = 1.0 - 2.0 * min(1.0, (row + 1) / n_lat)
    c_max = 1.0 - 2.0 * (row / n_lat)
    cos_t = c_min + v * (c_max - c_min)
    cos_t = torch.clamp(cos_t, -0.9999, 0.9999)
    sin_t = torch.sqrt(torch.clamp(1.0 - cos_t ** 2, min=1e-6))

    x = sin_t * torch.cos(phi)
    y = cos_t
    z = sin_t * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


class AtlasNet(nn.Module):
    """
    Complete Multi-View AtlasNet model with P surface patches and attention view pooling.
    """
    def __init__(
        self,
        num_patches: int = 25,
        latent_dim: int = 1024,
        hidden_dim: int = 256,
        pretrained: bool = True,
        template: str = "square",
        view_pooling: str = "max",
    ):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim = latent_dim
        self.template = template
        self.view_pooling = view_pooling
        self.coord_dim = 3 if template == "sphere" else 2
        self.arch_config = {
            "num_patches": num_patches,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "template": template,
            "view_pooling": view_pooling,
        }

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
        if self.view_pooling == "attention":
            self.view_attn = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

        # P MLP patch deformers
        self.deformers = nn.ModuleList([
            PatchDeformerMLP(latent_dim=latent_dim, hidden_dim=hidden_dim, coord_dim=self.coord_dim)
            for _ in range(num_patches)
        ])

    def encode_multiview(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-view features and pools across views to get a single global latent code z.
        """
        B, K, C, H, W = images.shape
        x_flat = images.view(B * K, C, H, W)
        feats = self.encoder_conv(x_flat)  # (B*K, 2048, 1, 1)
        feats = torch.flatten(feats, 1)    # (B*K, 2048)
        latents = self.fc_latent(feats)    # (B*K, latent_dim)

        latents_k = latents.view(B, K, self.latent_dim)
        if self.view_pooling == "attention":
            if K == 1:
                return latents_k[:, 0, :]
            attn_logits = self.view_attn(latents_k)  # (B, K, 1)
            weights = F.softmax(attn_logits, dim=1)
            z = (weights * latents_k).sum(dim=1)
            return z
        else:
            z, _ = torch.max(latents_k, dim=1)
            return z

    def sample_patch_points(
        self,
        points_per_patch: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Samples template coordinates for each patch.
        Returns: (num_patches, points_per_patch, coord_dim)
        """
        uv = torch.rand((self.num_patches, points_per_patch, 2), device=device)
        if self.template == "sphere":
            patches = [_uv_to_sphere_patch(uv[p], p, self.num_patches) for p in range(self.num_patches)]
            return torch.stack(patches, dim=0)
        return uv

    def forward(
        self,
        images: torch.Tensor,
        num_points: int = 4096
    ) -> Dict[str, torch.Tensor]:
        B = images.shape[0]
        device = images.device

        z = self.encode_multiview(images)  # (B, latent_dim)

        pts_per_patch = int(np.ceil(num_points / self.num_patches))
        patch_pts = self.sample_patch_points(pts_per_patch, device)  # (P, M, coord_dim)

        deformed_patches = []
        for p in range(self.num_patches):
            pts_in = patch_pts[p].unsqueeze(0).expand(B, -1, -1)
            pts_3d = self.deformers[p](pts_in, z)
            deformed_patches.append(pts_3d)

        all_points = torch.cat(deformed_patches, dim=1)
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
        self.eval()
        device = images.device
        with torch.no_grad():
            z = self.encode_multiview(images)

            u = torch.linspace(0, 1, grid_res, device=device)
            v = torch.linspace(0, 1, grid_res, device=device)
            grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
            grid_2d = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1).unsqueeze(0)

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
                coords = _uv_to_sphere_patch(grid_2d, p, self.num_patches) if self.template == "sphere" else grid_2d
                deformed = self.deformers[p](coords, z)
                v_np = deformed[0].cpu().numpy()
                all_verts.append(v_np)
                all_faces.append(patch_faces + vert_offset)
                vert_offset += v_np.shape[0]

            mesh_verts = np.concatenate(all_verts, axis=0)
            mesh_faces = np.concatenate(all_faces, axis=0)
            return mesh_verts, mesh_faces
