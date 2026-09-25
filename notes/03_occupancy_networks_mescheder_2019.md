# 03 — Occupancy Networks: beyond voxel grids

**Mescheder, Oechsle, Niemeyer, Nowozin, Geiger — "Occupancy Networks: Learning 3D
Reconstruction in Function Space" (CVPR 2019)** — arXiv:1812.03872

## Background

Voxels are a weak output parameterization: a 32³ grid caps us at **32k voxels**, and
the effective information per axis is tiny. **Occupancy Networks (ONet)** change the
*representation*:

```
encoder -> (latent code z per 3D location x)
f_theta(x, z) -> probability x is inside the object
```
- `f_theta` is a continuous function -> output can be sampled at ANY resolution
  (e.g., 128³, 256³) without re-training.
- Reconstruction is done by **isosurface extraction (marching cubes)** so you can
  produce watertight meshes.
- Distribution over the `inside/outside` is smooth, so the geometry is not
  staircased the way voxels are.

Related: 
- **DeepSDF** (Park et al. 2019): signed distance in function space; same idea, one
  latent code instead of per-location features.
- **IM-Net** (Chen & Zhang 2019): implicit decoder trained with occupancy + normals.

## Take-aways for our pipeline

- Our `32³` IoU@t=0.3 number is partly measuring **voxelization resolution**, not
  model quality. Even a perfect 32³ implied-silhouette can beat a great mesh, because
  the voxel GT dilates (see note 01) and the prediction is a hard-threshold field.
- ONet-style supervision needs GT we can infer occupancy at any voxel grid. Our
  point clouds would have to be **densified** (re-download point clouds at 16384
  pts, or run marching cubes from a fitted surface) to do this properly.
- **Action:** if you want to lift the ceiling, this is the architecture fix — but it
  requires dense/filled GT, which is expensive. First read note 06/07 (cheap wins).

## My notes

(add after reading)
