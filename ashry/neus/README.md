# Image-only NeuS-style reconstruction

This is a compact NeuS-style signed-distance-field baseline. It uses the same
20 input RGB renders and calibrated `transforms.json` cameras as the NeRF and
3DGS implementations. It does not load point clouds during training.

NeuS represents geometry as an SDF rather than unconstrained volume density. It
uses SDF-to-opacity conversion, a learned color network, an Eikonal regularizer,
and silhouette/RGB losses. The learned SDF can later be extracted with
marching cubes into an OBJ mesh.

Smoke test:

```bash
cd ashry/neus
../../.venv/bin/python train.py \
  --object battery/battery_001 --iterations 10 --image-size 64 \
  --rays-per-step 128 --coarse-samples 8 --fine-samples 8 \
  --output-dir results/battery_smoke
```

Recommended screening configuration:

```bash
../../.venv/bin/python train.py \
  --object battery/battery_001 --iterations 8000 --image-size 256 \
  --rays-per-step 2048 --coarse-samples 32 --fine-samples 32 \
  --output-dir results/battery_battery_001
```

## Export a mesh

NeuS has an explicit SDF, so extract its zero level set directly:

```bash
cd ashry/neus
../../.venv/bin/python extract_mesh.py \
  --checkpoint results/category_representatives_512_monitored/battery_battery_001/checkpoint.pt \
  --output results/category_representatives_512_monitored/battery_battery_001/mesh.obj \
  --resolution 192
```

The OBJ is extracted with marching cubes from the learned SDF. It is an
untextured geometry export; the checkpoint's neural color is not automatically
stored in the OBJ. Use the calibrated-view appearance tools separately if
you need colors or textures.
