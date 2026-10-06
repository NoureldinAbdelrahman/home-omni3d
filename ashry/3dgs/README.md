# 3D Gaussian Splatting

`train_3dgs.py` reconstructs an object from calibrated RGB images and
`transforms.json`. It does not use the ground-truth point cloud during
training. The point cloud is used later by shared evaluation tools.

```bash
PATH="../../.venv/bin:$PATH" ../../.venv/bin/python train_3dgs.py \
  --object battery/battery_001 \
  --image-size 256 \
  --iterations 10000 \
  --save-every 1000
```

The visual-hull initialization can be tuned with
`--visual-hull-candidates`, `--visual-hull-min-views`, and `--init-points`.

## Train multiple objects

Use the batch runner for a small validation sample first:

```bash
PATH="../../.venv/bin:$PATH" ../../.venv/bin/python train_all.py \
  --split val \
  --limit 5 \
  --iterations 10000 \
  --output-dir results/val_smoke
```

After checking those outputs, train the complete validation split:

```bash
PATH="../../.venv/bin:$PATH" ../../.venv/bin/python train_all.py \
  --split val \
  --iterations 10000 \
  --output-dir results/val
```

Use `--split train`, `--split test`, or `--split all` for the other frozen
object groups. Training is sequential because all objects share one GPU.
Existing checkpoints are skipped by default; use `--no-skip-existing` to
retrain them.

For a category-diverse sample with automatic visualizations:

```bash
PATH="../../.venv/bin:$PATH" ../../.venv/bin/python train_all.py \
  --split val \
  --one-per-category \
  --limit 10 \
  --iterations 10000 \
  --output-dir results/category_sample \
  --visualize
```

The selected qualitative comparisons are written to
`3dgs/qualitative/`.

## Export a mesh

The trained Gaussian representation is in `checkpoint.pt`. Convert it to an
approximate density-surface OBJ with:

```bash
cd ashry
../.venv/bin/python 3dgs/convert_mesh.py \
  --checkpoint 3dgs/results/category_representatives_512_monitored/battery_battery_001/checkpoint.pt \
  --output 3dgs/results/category_representatives_512_monitored/battery_battery_001/mesh.obj \
  --resolution 192 --level 0.08
```

This is an approximate mesh extracted from Gaussian density, not the native
Gaussian representation. The shared `tools/compare_nerf_3dgs_meshes.py` can
also create a corresponding NeRF mesh and appearance exports.
