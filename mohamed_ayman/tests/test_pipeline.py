"""
Comprehensive Sanity and Unit Test Suite for Multi-View 3D Reconstruction Pipeline.
Tests:
- Point cloud normalization & voxelization
- Marching Cubes mesh generation
- OmniObject3DDataset multi-view loading
- Pix2Vox multi-view forward & backward pass
- AtlasNet multi-view forward & backward pass + mesh extraction
- Chamfer Distance, F-Score, and Voxel IoU calculation
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.voxelizer import normalize_point_cloud, points_to_voxels, voxels_to_mesh
from src.data.splits import create_or_load_splits
from src.data.dataset import OmniObject3DDataset
from src.losses.chamfer_distance import chamfer_distance, compute_f_score
from src.losses.voxel_losses import Pix2VoxLoss, voxel_iou
from src.models.pix2vox import Pix2Vox
from src.models.atlasnet import AtlasNet


class TestReconstructionPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_voxelizer_and_mesh(self):
        # Synthetic unit sphere point cloud
        phi = np.random.uniform(0, 2 * np.pi, 1000)
        costheta = np.random.uniform(-1, 1, 1000)
        theta = np.arccos(costheta)
        r = 50.0  # mm
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        raw_pc = np.stack([x, y, z], axis=-1)

        norm_pc, center, max_dist = normalize_point_cloud(raw_pc)
        self.assertAlmostEqual(center[0], 0.0, delta=5.0)
        self.assertLessEqual(np.max(np.abs(norm_pc)), 0.5)

        vox = points_to_voxels(norm_pc, voxel_res=32, dilate=True)
        self.assertEqual(vox.shape, (32, 32, 32))
        self.assertGreater(vox.sum(), 0)

        mesh = voxels_to_mesh(vox, threshold=0.5)
        self.assertIsNotNone(mesh)
        verts, faces = mesh
        self.assertGreater(verts.shape[0], 0)
        self.assertEqual(faces.shape[1], 3)
        print("✓ Voxelizer and Marching Cubes test passed.")

    def test_losses_and_metrics(self):
        p1 = torch.zeros((2, 100, 3), device=self.device)
        p2 = torch.zeros((2, 100, 3), device=self.device)
        cd, _ = chamfer_distance(p1, p2, distance_type="l2")
        self.assertAlmostEqual(cd.item(), 0.0, places=5)

        f1, prec, rec = compute_f_score(p1, p2, threshold=0.01)
        self.assertAlmostEqual(f1.mean().item(), 1.0, places=3)

        v1 = torch.ones((2, 32, 32, 32), device=self.device)
        v2 = torch.ones((2, 32, 32, 32), device=self.device)
        iou_mean, _ = voxel_iou(v1, v2, threshold=0.5)
        self.assertAlmostEqual(iou_mean.item(), 1.0, places=3)
        print("✓ Chamfer distance, F-score, and IoU metric tests passed.")

    def test_pix2vox_forward_backward(self):
        model = Pix2Vox(pretrained=False, use_refiner=True).to(self.device)
        criterion = Pix2VoxLoss(bce_weight=1.0, dice_weight=0.5)

        B, K = 2, 3
        dummy_images = torch.randn(B, K, 3, 224, 224, device=self.device)
        dummy_gt = (torch.rand(B, 32, 32, 32, device=self.device) > 0.8).float()

        out = model(dummy_images)
        self.assertEqual(out["voxels"].shape, (B, 32, 32, 32))
        self.assertEqual(out["coarse_voxels"].shape, (B, 32, 32, 32))

        loss = criterion(out["voxels"], dummy_gt, out["coarse_voxels"])
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        print("✓ Pix2Vox multi-view (K=3) forward & backward pass passed.")

    def test_atlasnet_forward_backward(self):
        model = AtlasNet(num_patches=10, latent_dim=256, hidden_dim=64, pretrained=False).to(self.device)
        B, K = 2, 3
        dummy_images = torch.randn(B, K, 3, 224, 224, device=self.device)
        dummy_gt = torch.randn(B, 1000, 3, device=self.device)

        out = model(dummy_images, num_points=1000)
        self.assertEqual(out["points"].shape, (B, 1000, 3))

        loss, _ = chamfer_distance(out["points"], dummy_gt)
        self.assertFalse(torch.isnan(loss))
        loss.backward()

        verts, faces = model.generate_mesh(dummy_images[:1], grid_res=6)
        self.assertGreater(verts.shape[0], 0)
        self.assertEqual(faces.shape[1], 3)
        print("✓ AtlasNet multi-view (K=3) forward, backward & mesh extraction passed.")

    def test_dataset_loading(self):
        if os.path.exists("dataset/renders"):
            splits = create_or_load_splits(dataset_root="dataset")
            ds = OmniObject3DDataset(samples=splits["train"][:5], dataset_root="dataset", num_views=3, is_train=False)
            sample = ds[0]
            self.assertEqual(sample["images"].shape, (3, 3, 224, 224))
            self.assertEqual(sample["point_cloud"].shape, (4096, 3))
            self.assertEqual(sample["voxels"].shape, (32, 32, 32))
            print(f"✓ Real OmniObject3D dataset sample loaded successfully: {sample['category']}/{sample['object_id']}")


if __name__ == "__main__":
    unittest.main()
