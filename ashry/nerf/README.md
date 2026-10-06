# NVIDIA NeRF-style image-only reconstruction

This folder contains a calibrated multi-view NeRF baseline. It uses only the
20 RGB input renders and `transforms.json`; point clouds are not loaded.
Camera poses use the same NeRF/OpenGL convention as the dataset.

## Smoke test

```bash
cd ashry/nerf
../../.venv/bin/python train.py \
  --object battery/battery_001 --iterations 1000 --image-size 256 \
  --rays-per-step 2048 --coarse-samples 48 --fine-samples 48 \
  --output-dir results/battery_smoke
```

The trainer writes a checkpoint and held-out renders for views `020`-`023`.
For a category-balanced run:

```bash
../../.venv/bin/python train_all.py \
  --one-per-category --limit 10 --iterations 8000 --image-size 256 \
  --rays-per-step 2048 --coarse-samples 32 --fine-samples 32 \
  --skip-existing
```

The batch script defaults to this faster configuration. It is intended for
screening many objects; use 512x512 with 64+64 samples for final high-quality
runs on selected objects. `--skip-existing` allows an interrupted batch to
resume without retraining completed objects.

The 1024x1024 source renders are downsampled only for training when
`--image-size 512` is used. Set `--image-size 1024` when full-resolution
training and held-out rendering are required. The upgraded trainer uses hierarchical
coarse/fine sampling, foreground-biased rays, and a silhouette loss. Rendering
can use `--image-size 1024` after training; `--render-chunk` limits VRAM use.
These settings are designed for a 16 GB RTX 5070 Ti and 32 GB system RAM.

Mesh exports can include a UV-textured OBJ/MTL/PNG set generated from the
original calibrated RGB views. Prefer those textured exports over vertex-colored
PLY files when inspecting fine labels or printed details.

This is a compact NeRF baseline implemented in current PyTorch rather than an
old CUDA-specific NVIDIA repository. It is image-only at reconstruction time
and is intended to be compared with the existing image-only 3DGS pipeline.

## Export a mesh

Use the shared comparison/export script. It evaluates the trained NeRF density
on a 3D grid and extracts an approximate surface with marching cubes:

```bash
cd ashry/tools
../../.venv/bin/python compare_nerf_3dgs_meshes.py \
  --object battery/battery_001 \
  --nerf-results ../nerf/results/category_representatives_512_monitored \
  --gs-results ../3dgs/results/category_representatives_512_monitored \
  --output-dir qualitative/category_representatives_512_meshes \
  --resolution 192
```

The output contains the NeRF OBJ and comparison artifacts. The mesh is an
isosurface of predicted density, so it is an approximate geometry export, not
the native NeRF representation. The same command can produce a UV-atlas
textured OBJ from the original calibrated views; inspect its README output
notes because per-face camera texturing can show seams.
