# 00 — Single-View 3D Reconstruction: the field and our metrics

## Background

The task set is: **one 224-view render → a complete 3D shape** of a household object.
This belongs to the *single-view 3D reconstruction (SVR)* family. Follow the paper
trail in order; each builds on the last.

| Paper | What it established |
|---|---|
| `Shape from Silhouette` (Laurentini 1994) | visual-hull / occupancy idea |
| `3D Shape Perception from Monocular Vision` (Saxena, Sun, Ng ~2009) | depth + shape priors, discriminative exits |
| Lucas et al., **`View-Invariant Deep Shape Prediction`** (CVPR 2014) | CNN → shape prior (not pose-specific) |
| Kar et al., **`Category-Specific Object Reconstruction from a Single Image`** (CVPR 2015) | silhouette-then-reconstruct;
 used in Choy's paper |
| `Learning a ConvNet for Morphable-Shape prediction` (Hu et al. 2017) — optional | shape parameterized as a fixed category template + deformation |

Metrics used by every SVR paper (we need to align with them):

- **IoU** on a voxel grid, reported at different binarization thresholds `t`.
- **Chamfer Distance (CD)** between predicted and expected point clouds/meshes.
- **F-Score@τ** (τ = distance tolerance) from Chamfer's two distances.

## Take-aways for our pipeline

- Our task is architecture-agnostic to start; the Baseline numbers (Pix2Vox,
  AtlasNet-SVR, NeRF oracle) that we have now are all in the same family and can be
  improved one at a time.
- The headline metric should always be reported at **multiple thresholds**
  (0.2/0.3/0.4/0.5) *and* per-category, because of our long tail.
- If a paper reports one threshold, treat its number as an exact-protocol reference
  and not a head-to-head.

## My notes

(add after reading)
