# Bones - COLMAP, Additive-Subtractive Modeling

## COLMAP (multi-view, classical)

Sparse reconstruction from the 24 renders per object with pycolmap (CPU),
scored against the GT point clouds. Multi-view reference, like the NeRF oracle:
not directly comparable to the single-image models.

### Setup

```bash
conda create -n colmap python=3.11 && conda activate colmap
pip install pycolmap==4.2.1 numpy scipy pillow matplotlib open3d
```

Needs `dataset/` with **fixed** point-cloud names (`python fix_point_cloud_names.py`).

### Pipeline

| Script | What it does |
|---|---|
| `gt_align.py` | Fits each GT cloud into the render/camera frame (rotation + scale + offset) against the 24 alpha silhouettes under the GT cameras. Writes `bones/gt_alignment.json`. Fits with < 90% of points inside the silhouettes are marked unreliable and not scored. |
| `colmap_sparse.py` | Masked SIFT + exhaustive matching, then `--mode known` (triangulate with GT cameras) or `--mode sfm` (blind incremental SfM, Sim(3)-aligned to GT cameras for pose error and for the points). Points in the render frame go to `bones/work/<mode>/<cat>__<obj>/points.npy`; per-object stats to `results/colmap/sparse_<mode>/objects.json`. |
| `eval_clouds.py` | Normalises both clouds by the GT unit sphere; Chamfer-L1 and precision/recall/F at 1%, 2%, 5%, plus the per-object F ceiling. Writes `metrics.csv` + `summary.json`. |

```bash
python bones/colmap_sparse.py --mode known --per-category 4   # 4 random objects per category
python bones/gt_align.py --objects <same objects>              # or no args for all objects (slow, ~1.5 min/object/core)
python bones/eval_clouds.py results/colmap/sparse_known
```

`bones/work/` (COLMAP databases, sparse models, points) is gitignored.

### 3D viewer

Results are exported to the team viewer (`viewer/`, see viewer/README.md):

```bash
python bones/export_colmap_to_viewer.py   # -> viewer/runs/bones__colmap__sparse_{known,sfm}/
python viewer/view.py                     # Ctrl+C to stop
```

### Reading the numbers

- GT clouds have 4096 points, ~1.6% of the object radius apart, so even a
  perfect surface cannot reach F = 1 at τ = 1%. `f_ceiling_<τ>` is the GT scored
  against itself (leave-one-out): ~0.28 at 1%, ~0.67 at 2%, ~0.99 at 5%.
- Sparse SIFT points are accurate (high precision) but only cover textured
  regions (low recall); plain or shiny objects (pan, bowl, cup) get few points.
- `known` is the best case (all 24 views posed); `sfm` shows what COLMAP
  recovers on its own (views often only partly registered, wide baselines).
