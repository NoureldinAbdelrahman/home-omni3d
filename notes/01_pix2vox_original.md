# 01 — Pix2Vox: Original Paper

**Xie, Zhang, Zheng, Liu, Fang — "Pix2Vox: Context-Aware Complete 3D Reconstruction
from a Single Image" (CVPR 2019)** — arXiv:1901.11153

## Background

The paper introduced the network our baseline uses, and everything below matches what
is in `Pix2Vox/`.

Architecture:
```
input 224x224 RGB
        |
   Encoder (VGG16-BN, ImageNet pretrained) → 512-d feature map
        |
   Decoder — 3 conv2d → FC ×2 → conv3d / 3D transposed convs
        |
   candidate reconstruction at high and low resolution  (4 candidates!)
        |
   Merger  (soft attention over the candidates)
        |
   coarse 32³ volume
        |
   Refiner (3D convolutions) -> final 32³ volume
        |
   BCE loss × 10 (against the voxelized ground truth)
```

Protocol in the paper:
- ShapeNet-13 (13 categories, ~35k models). Train ~250 epochs.
- IoU reported **per category** at a fixed threshold; best threshold from validation.
- Number of input views: **1 view** (Pix2Vox-Large uses 5 views and reports better IoU).

Key implementation details that materially affect the score:
1. `USE_MERGER` and `USE_REFINER` are **both on from epoch 1** in this repo
  (`EPOCH_START_USE_REFINER = 0`, `EPOCH_START_USE_MERGER = 0`).
2. Voxelization of the ground truth matters enormously; the paper's GT for ShapeNet
  is watertight voxel occupancy, while our `points_to_voxel` **dilates a point
  cloud by 1 voxel** to approximate surface occupancy. That dilation puts many more
  voxels in the *expected* GT and punishes a "thin shell" prediction less — one
  reason our IoU is lower than published.

## Take-aways for our pipeline

- Our setup differences vs the paper that could explain part of the gap:
  1. **GT voxelization is stricter / sparser** than watertight mesh occupancy.
  2. Class-imbalance: 31/69 classes have <3 objects. Our `Macro` column *is* the
     per-category mean the paper uses, and is much lower than micro.
  3. `BATCH_SIZE` of 8 (paper defaults to larger for 3D convs memory).
- **Action:** repeat with `N_VIEWS_RENDERING > 1` (Paper's `Pix2Vox-A` shows 3-view
  fusion), because our `--n-views` flag supports it now.

## My notes

(add after reading)
