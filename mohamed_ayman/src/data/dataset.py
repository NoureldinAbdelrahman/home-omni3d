"""
OmniObject3D Dataset for Multi-View 3D Shape Reconstruction.
Handles multi-view RGB image loading with background compositing,
canonical point cloud normalization, and cached 32x32x32 voxel occupancy grids.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.data.voxelizer import normalize_point_cloud, points_to_voxels
from src.data.splits import find_dataset_root


class OmniObject3DDataset(Dataset):
    """
    Multi-View Dataset for OmniObject3D objects.
    """
    def __init__(
        self,
        samples: List[Dict[str, str]],
        dataset_root: str = None,
        num_views: int = 3,
        image_size: int = 224,
        is_train: bool = True,
        voxel_res: int = 32,
        cache_voxels: bool = True
    ):
        super().__init__()
        self.samples = samples
        self.dataset_root = find_dataset_root(dataset_root)
        self.num_views = num_views
        self.image_size = image_size
        self.is_train = is_train
        self.voxel_res = voxel_res
        self.cache_voxels = cache_voxels

        self.renders_dir = os.path.join(self.dataset_root, "renders")
        self.pcs_dir = os.path.join(self.dataset_root, "point_clouds")
        self.cache_dir = os.path.join(self.dataset_root, "cache", f"voxels_{voxel_res}")
        if self.cache_voxels:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _select_view_indices(self, total_available: int = 24) -> List[int]:
        if self.is_train:
            # Random selection during training for viewpoint diversity
            indices = np.random.choice(total_available, size=self.num_views, replace=False)
            return sorted(indices.tolist())
        else:
            # Deterministic equidistant view selection during evaluation
            step = total_available / max(1, self.num_views)
            return [int(round(i * step)) % total_available for i in range(self.num_views)]

    def _load_image(self, img_path: str) -> torch.Tensor:
        with Image.open(img_path) as im:
            im = im.convert("RGBA")
            # Composite onto white background
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, im).convert("RGB")
            
            if composite.size != (self.image_size, self.image_size):
                composite = composite.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR
                )
            
            # To tensor and normalize
            arr = np.array(composite, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            return self.normalize_transform(t)

    def _get_voxels(self, cat: str, obj_id: str, pc_norm: np.ndarray) -> np.ndarray:
        if self.cache_voxels:
            cache_file = os.path.join(self.cache_dir, f"{cat}_{obj_id}.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
            
            voxels = points_to_voxels(pc_norm, voxel_res=self.voxel_res, dilate=True)
            np.save(cache_file, voxels)
            return voxels
        else:
            return points_to_voxels(pc_norm, voxel_res=self.voxel_res, dilate=True)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        sample = self.samples[idx]
        cat = sample["category"]
        obj_id = sample["object_id"]

        obj_render_dir = os.path.join(self.renders_dir, cat, obj_id)
        pc_path = os.path.join(self.pcs_dir, cat, f"{obj_id}.npy")

        # 1. Load multi-view images
        view_indices = self._select_view_indices(total_available=24)
        image_tensors = []
        for v in view_indices:
            v_name = f"{v:03d}.png"
            img_file = os.path.join(obj_render_dir, v_name)
            if not os.path.exists(img_file):
                # Fallback to 000.png if specific index missing
                img_file = os.path.join(obj_render_dir, "000.png")
            image_tensors.append(self._load_image(img_file))

        # Shape: (K, 3, H, W)
        images = torch.stack(image_tensors, dim=0)

        # 2. Load and normalize Point Cloud
        raw_pc = np.load(pc_path)  # (4096, 3)
        norm_pc, centroid, scale = normalize_point_cloud(raw_pc)
        point_cloud = torch.from_numpy(norm_pc).float()

        # 3. Load or generate Ground Truth 32^3 Voxels
        voxels_np = self._get_voxels(cat, obj_id, norm_pc)
        voxels = torch.from_numpy(voxels_np).float()  # (32, 32, 32)

        return {
            "images": images,                # (K, 3, H, W)
            "point_cloud": point_cloud,      # (N, 3)
            "voxels": voxels,                # (32, 32, 32)
            "category": cat,
            "object_id": obj_id,
            "scale": scale
        }
