# Team 3D viewer

Look at every model's predicted point clouds next to the ground truth, rotate
them in 3D, and compare scores per object and per category.

```bash
python viewer/view.py        # opens http://localhost:8765 ; Ctrl+C to stop
```

Needs `numpy`, `scipy`, `pillow` (already in `requirements.txt`) and the shared
`dataset/` with **fixed** point-cloud names (`python fix_point_cloud_names.py`).
GT clouds and photos are read from `dataset/`; only predictions are stored in
the repo. On WSL the page opens in your Windows browser. Options:
`--port 9000`, `--no-browser`, `--export DIR` (static copy you can host anywhere).

## Add your results

Each run is one folder under `viewer/runs/<owner>__<model>__<name>/`. Write it
from your own evaluation code, then commit it on your branch:

```python
import sys; sys.path.insert(0, "viewer")       # from the repo root
from export import ViewerRun, normalize_like_gt, voxels_to_points

run = ViewerRun(owner="nesegemaa", model="AtlasNet", name="svr25",
                label="AtlasNet SVR, 25 patches", notes="test split, input view 0")
for obj_id, pts in predictions.items():         # obj_id like "chair/chair_001"
    run.add(obj_id, pts)                         # (N, 3) in the shared frame (below)
run.save()
```

`save()` scores every object against the GT (same code for everyone) and
prints the run's mean F@5%. Re-saving the same owner/model/name replaces it.

### The shared frame

All runs are drawn and scored in the GT's unit sphere: GT cloud from
`dataset/point_clouds/<cat>/<obj>.npy`, centred on its bounding-box centre,
scaled so the farthest GT point is at distance 1 (docs/METRICS.md).

| Your output | Call |
|---|---|
| Points already in that frame | `run.add(obj_id, pts)` |
| Points in the raw `dataset/point_clouds` frame | `run.add(obj_id, pts, frame="raw")` |
| Pix2Vox voxels (grid from `prepare_model_data.points_to_voxel`) | `run.add(obj_id, voxels_to_points(vox, obj_id, threshold=0.3))` |
| Your own normalisation | undo it to the raw frame first, then `frame="raw"` |

Wrong frame is the usual mistake: if your cloud floats away from the grey GT in
the viewer, check the normalisation. Single-view models trained on
`dataset/point_clouds` usually predict in a centred/scaled version of the raw
frame, so undo that scaling and pass `frame="raw"`.

### Optional per-object extras

```python
run.add(obj_id, pts,
        info={"input view": 0},            # shown as rows in the score table
        cameras=cams, cameras_used=used,   # (V,3) camera centres + flags, drawn as cones
        excluded="GT fit unreliable")      # shown but not scored
```

## What the numbers mean

Chamfer-L1 (mean nearest-neighbour distance both ways) and precision / recall /
F at 1%, 2%, 5% of the radius. The GT has 4096 points (~1.6% apart), so a
perfect surface still scores only ~0.28 at 1%; the viewer shows that ceiling per
object. Runs can cover different objects: compare them on shared objects.

## Files

| File | Role |
|---|---|
| `view.py` | Local server (foreground, Ctrl+C), static `--export` |
| `export.py` | `ViewerRun`, frame helpers, the shared `score()` |
| `index.html` | The page (three.js from a CDN) |
| `runs/` | One folder per run: `run.json` (metadata, scores) + `pts/<cat>.json` |
| `screenshot.py` | Headless screenshots for checking the page (needs playwright) |
