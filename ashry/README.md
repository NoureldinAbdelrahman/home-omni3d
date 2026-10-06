# Reconstruction experiments

This directory contains the reproducible image-to-3D experiments for
OmniObject3D. The dataset is an external symlink and is never committed.

## Layout

- `benchmark/` - frozen object splits, camera-view protocol, and evaluation metadata.
- `tools/` - shared visualization and inspection utilities.
- `3dgs/` - image-only 3D Gaussian Splatting reconstruction.
- `nerf/` - NVIDIA NeRF implementation/results area.
- `neus/` - image-only NeuS-style signed-distance reconstruction.

Each method owns its code and generated outputs. The benchmark point clouds are
used for evaluation only.

## Run all three methods overnight

This coordinator trains 3DGS, NeRF, and NeuS at 512x512, one CUDA process at a
time:

```bash
cd ashry
../../.venv/bin/python train_all_models.py --split val
```

It skips existing `checkpoint.pt` files, continues after failed objects, and
writes subprocess output to `results/overnight_training.log`. Use
`--one-per-category` or `--limit 1` for a smaller test run. The defaults are
screening settings: 6,000 3DGS iterations, 1,024 rays with 24+24 NeRF
samples, and 1,024 rays with 32+32 NeuS samples.

The coordinator is sequential by design. Running multiple trainers on the same
16 GB GPU generally causes VRAM contention and makes each process slower. Only
run independent coordinators in parallel when assigning them to separate GPUs.

## Freeze or verify the benchmark

```bash
cd ashry/benchmark
../.venv/bin/python freeze_benchmark.py
```

The manifest uses input views `000`-`019` and held-out views `020`-`023`.

## Run image-only 3DGS

```bash
cd ashry/3dgs
PATH="../../.venv/bin:$PATH" ../../.venv/bin/python train_3dgs.py \
  --object battery/battery_001 \
  --image-size 256 \
  --iterations 10000 \
  --save-every 1000
```

The trainer uses only the 20 input images and `transforms.json`. It initializes
Gaussians from image silhouettes, then learns appearance and geometry with
densification and pruning. The target point cloud is not loaded.

## Run image-only NeuS

```bash
cd ashry/neus
../../.venv/bin/python train.py \
  --object battery/battery_001 \
  --iterations 8000 --image-size 256 \
  --rays-per-step 2048 --coarse-samples 32 --fine-samples 32
```

NeuS uses the same RGB input views and calibrated cameras, but represents
geometry as a signed-distance field. Its checkpoint can be converted to a mesh
with a later density/SDF-grid extraction step.

Outputs are written to `3dgs/results/battery_battery_001/`. The checkpoint is
the complete Gaussian model; `gaussian_centers.npy` is an evaluation export.

## Visualize a reconstruction

```bash
cd ashry/tools
../../.venv/bin/python visualize_object.py \
  --object battery/battery_001 \
  --prediction ../3dgs/results/battery_battery_001/gaussian_centers.npy \
  --render-dir ../3dgs/results/battery_battery_001/heldout
```
