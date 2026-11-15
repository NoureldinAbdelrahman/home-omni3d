# Viewer run index

One folder per run: `<owner>__<model>__<name>/` with `run.json` (metadata + scores) and `pts/<cat>.json` (predictions). Folder names are stable identifiers cited by the deck and outline: do not rename them.

Each run is committed on its owner's branch. `main` carries only the two COLMAP runs (promoted with the viewer). To see all runs, check out the branch (or merge its `viewer/runs/` locally) and run `python viewer/view.py`.

All scores below are the viewer's own scorer (`viewer/export.py::score`): GT bounding-box frame, farthest GT point at 1, 4,096 points per object, Euclidean CD-L1, F at 1/2/5% of the radius. Micro means from each `run.json` `summary`. **ICP-aligned runs use the GT to fix rotation: an oracle upper bound for display, not a fair score.**

| Run ID | Branch | Owner | Model | Protocol | Objects | Raw / aligned | CD-L1 | F@5% | Cited in MS2 |
|---|---|---|---|---|---:|---|---:|---:|---|
| `bones__colmap__sparse_known` | `bones`, `main` (also copied to `nesegemaa`, `mohamed-ayman`) | Noureldeen A. | COLMAP | 24 views + GT cameras | 294 (286 scored) | raw (GT frame via cameras) | 0.134 | 0.539 | yes, slides 9, 19, 22 (values via `results/colmap/*/summary.json`) |
| `bones__colmap__sparse_sfm` | same | Noureldeen A. | COLMAP | 24 views, blind SfM, Sim(3) to GT cameras | 294 (286 scored) | raw | 0.371 | 0.378 | yes, slide 19 |
| `mohamed_ayman__pix2vox__base` | `mohamed-ayman` | Mohamed A. | Pix2Vox | 3 views, 32³ voxels → points | 294 (COLMAP objects; only 48 in his test split) | raw | 0.226 | 0.300 | yes, slide 18 (gallery only) |
| `mohamed_ayman__pix2voxplusplus__refined` | `mohamed-ayman` | Mohamed A. | Pix2Vox++ | as above, refined | 294 (48 in test split) | raw | 0.227 | 0.302 | yes, slide 18 (gallery only) |
| `nesegemaa__atlasnet__a2_p25_sphere_s123` | `nesegemaa` | Noureldin S. | AtlasNet-SVR | 1 view, team test split, SPHERE winner, shown at 4,096 of 64,050 pts | 260 | raw | 0.226 | 0.196 | yes, slides 13, 14, 20 |
| `nesegemaa__atlasnet__a2_p25_sphere_ft1e4` | `nesegemaa` | Noureldin S. | AtlasNet-SVR | low-LR finetune of the winner | 260 | raw | 0.223 | 0.196 | no |
| `nesegemaa__triposr__default_masked_full` | `nesegemaa` | Noureldin S. | TripoSR | 1 view, masked, team test split | 260 | raw | 0.404 | 0.163 | yes, slide 13 |
| `nesegemaa__triposr__default_masked_full_aligned` | `nesegemaa` | Noureldin S. | TripoSR | as above | 260 | **ICP-aligned** | 0.275 | 0.251 | yes, slides 13, 14 (display), 20 |
| `nesegemaa__point_e__default_masked_g3_full` | `nesegemaa` | Noureldin S. | Point-E | 1 view, masked, guidance 3, team test split | 260 | raw | 0.491 | 0.142 | yes, slide 13 |
| `nesegemaa__point_e__default_masked_g3_full_aligned` | `nesegemaa` | Noureldin S. | Point-E | as above | 260 | **ICP-aligned** | 0.226 | 0.349 | yes, slides 13, 14 (display), 20 |
| `nesegemaa__triposr__default_masked` | `nesegemaa` | Noureldin S. | TripoSR | 5-object panel | 5 | raw | 0.338 | 0.197 | no |
| `nesegemaa__triposr__default_masked_aligned` | `nesegemaa` | Noureldin S. | TripoSR | 5-object panel | 5 | **ICP-aligned** | 0.309 | 0.216 | no |
| `nesegemaa__point_e__default_masked_g3` | `nesegemaa` | Noureldin S. | Point-E | 5-object panel | 5 | raw | 0.509 | 0.141 | no |
| `nesegemaa__point_e__default_masked_g3_aligned` | `nesegemaa` | Noureldin S. | Point-E | 5-object panel | 5 | **ICP-aligned** | 0.252 | 0.354 | no |
| `nesegemaa__point_e__p1_best_3views` | `nesegemaa` | Noureldin S. | Point-E | 5-object panel, best of views 0/8/16 (**not single-view**) | 5 | raw | 0.500 | 0.142 | no |
| `nesegemaa__point_e__p3_raw` | `nesegemaa` | Noureldin S. | Point-E | 5-object panel, raw RGB input | 5 | raw | 0.495 | 0.153 | no |

Size: about 62 MB of predictions on `nesegemaa` (14 runs), 31 MB on `mohamed-ayman` (4 runs), 9 MB on `main` (2 runs).

Caveats:

- The Pix2Vox exports cover the 294 COLMAP objects, and only 48 are in Mohamed A.'s test split: the other 246 were seen in training. Their viewer means are **not** test scores. Use the evaluation reports ([`results/README.md`](../../results/README.md)).
- The viewer's F@1% ceiling is ≈ 0.27 for a perfect 4,096-point prediction; use F@5% ([`docs/METRICS.md`](../../docs/METRICS.md)).
- Runs cover different objects: compare runs only on shared objects.
