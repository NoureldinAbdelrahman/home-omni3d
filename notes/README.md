# Papers Reading Index

`.md` files in this folder map **one paper to one measured bottleneck** in our
pipeline. Read in the suggested order; each file has a summary/what-to-take-away
section (`## Take-aways for our pipeline`) at the end. When you have read one,
add notes under `## My notes` and we re-run what that paper justifies.

| # | File | Topic | Fixes in our code |
|---|------|-------|-------------------|
| 0 | `00_single-view-reconstruction-foundations.md` | the field + metrics | — |
| 1 | `01_pix2vox_original.md` | architecture, IoU@t, merger/refiner | reproduce protocol exactly |
| 2 | `02_3d_shapes_from_single_image_choy_2016.md` | 3D-R2N2, LSTM multi-view | `--n-views`, generative capacity |
| 3 | 3 `03_occupancy_networks_mescheder_2019.md` | why 32³ voxels cap quality | occupancy networks / continuous output |
| 4 | `04_im2avatar_and_point_canonicalization.md` | non-rigid deformation (AtlasNet's core idea) | num primitives, patch mapping |
| 5 | `05_fscore_ciagar_2019_chamfer_pitfalls.md` | **how to read our Chamfer/F correctly** | evaluation setup only |
| 6 | `06_long_tail_classification_balanced.md` | long-tailed training (31/69 cats <3) | `--max-per-class`, class weighting |
| 7 | `07_resnet_pretraining_transfer_views.md` | **pretrained image encoder** (AtlasNet's worst bug) | pretrained encoder flag |
| 8 | `08_single-view_shape_diffusion_and_gpt.md` | modern generative priors | optional stretch |
| 9 | `09_nerf_object_benchmarks_and_oracle.md` | our multi-view oracle design | rename/oracle row |

## How to use

> New to the numbers? Read [`docs/METRICS.md`](../docs/METRICS.md) first — it
> explains IoU thresholds, Chamfer vs F-score, PSNR/SSIM, and micro-vs-macro in
> plain language.

1. Read the file.
2. Scroll to the bottom, append `## My notes` and (optionally) `## Do next`.
3. Tell me (or push). I re-run the corresponding experiment/training and update
   `results/`.

Do not edit the `Background` sections; keep them stable so new notes are easy to diff.
