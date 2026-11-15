# Metrics: definitions, scales, comparability

Plain-language reference for every metric the team reports. Current numbers and their source files are in [`results/README.md`](../results/README.md). Pipeline: [`ROADMAP.md`](ROADMAP.md).

Always report Chamfer **with** F-score, and micro **with** macro. Flag categories with fewer than 3 test objects as noisy (35 of the 76 categories on the team split).

- *micro* = mean over all test objects (big classes dominate).
- *macro* = mean of per-category means (every category counts equally).

## 1. IoU at t = 0.2 / 0.3 / 0.4 / 0.5 (voxels: Pix2Vox, Pix2Vox++)

The model outputs a 0–1 occupancy score for each of the 32³ cells. A threshold t turns it into a solid shape.

```
IoU = cells solid in BOTH prediction and GT / cells solid in EITHER
```

1.0 is perfect overlap. Four thresholds stop a model from looking good by predicting oversized shapes (helps at low t) or undersized ones (helps at high t). The MS2 headline is **IoU@0.4**.

## 2. Chamfer + F-score (surfaces)

**Chamfer**: for each predicted point, the distance to the nearest GT point, averaged; plus the reverse direction. Lower is better. It is an average, so a diffuse cloud that covers the object can score well without having the right shape.

**F-score@τ**: precision = share of predicted points within τ of the GT; recall = share of GT points within τ of the prediction; F = 2PR/(P+R). It catches blobs that Chamfer hides.

Both depend on **point density**: a denser prediction gets a lower Chamfer and a different F. Compare runs only at matched point counts.

### The scales in use

The same names mean different things on different tracks. Always say which one.

| Name | Used by | Frame | Distance | τ | Points |
|---|---|---|---|---|---|
| Chamfer, **F@τ=0.001** | AtlasNet-SVR, Point-E, TripoSR (`nesegemaa`, `results/atlasnet/*/qual_agg.json`) | UnitBall-normalised clouds | **squared** Euclidean | τ² = 0.001, ≈ 0.032 Euclidean (`tau_euclid_approx`) | prediction as output (AtlasNet SPHERE: 64,050; SQUARE: 2,500; priors: 4,096) vs 4,096 GT |
| **CD-L1, CD-L2, F@0.01, F@0.02** | Pix2Vox, Pix2Vox++ (`mohamed_ayman/outputs/evaluations/`) | the track's own frame | L1 / L2 | 0.01, 0.02 | points from 32³ voxels |
| **CD-L1, F@1/2/5%** | team viewer (`viewer/export.py::score`), COLMAP (`bones/eval_clouds.py`) | GT bounding-box centre, farthest GT point at 1 | Euclidean, mean both ways | 1%, 2%, 5% of the radius | 4,096 per object (COLMAP ≈ 1,900) |
| **CD-L2, F1@0.01, F1@0.02** | 3DGS (`ashry/3dgs_metrics/`) | GT unit sphere × 0.4; prediction scale- and centre-aligned to GT bounding box | squared (CD-L2) | 0.02 = 5% of the 0.4 radius | 4,096 Gaussian centres |

Notes:

- **F@1% has a low ceiling.** The 4,096-point GT is about 1.6% apart, so even a perfect prediction scores ≈ 0.27 at 1% (viewer `f_ceiling_1` 0.276, `f_ceiling_5` 0.992 in `bones:results/colmap/sparse_known/summary.json`). F@5% is the useful viewer scale.
- **MS1 wording.** Earlier versions of this file called the AtlasNet F-score "F@1% (τ = 0.01)". The code uses `threshold=0.001` on squared distances (`milestone-1:AtlasNet/auxiliary/ChamferDistancePytorch/fscore.py`), i.e. about 3% of the radius. MS1 AtlasNet F-scores (0.079–0.091) use that scale.
- **Historical MS1 example.** MS1 versions of this file quoted `cup` with Chamfer 5.84 and F ≈ 0.0006 from an early MS1 AtlasNet run. It is not comparable with any MS2 number.
- **best vs final epoch.** *best* = best validation epoch (the model that is deployed); *final* = last epoch. AtlasNet final epochs can explode (up to 71,000× worse), so only best-epoch weights are scored.

### AtlasNet SPHERE point density

The SPHERE template always outputs 64,050 points (25 × 2,562), against 2,500 for the SQUARE baseline. Scores for the winning run `a2_p25_sphere_s123` (`nesegemaa:nesegemaa/PAPER.md` §6.1 audit note, `results/atlasnet/a2_p25_sphere_s123/qual_agg.json`):

| Points | Chamfer | F@τ=0.001 |
|---|---:|---:|
| 64,050 (as output) | 0.0332 | 0.161 |
| 4,096 (subsampled, matches the priors) | 0.0384 | 0.090 |
| 2,500 (matches the SQUARE baseline) | 0.0409 | 0.073 |

Baseline `a0_svr25_imgnet` at 2,500 points: 0.0467 / 0.083. At matched density the Chamfer gain is ~12%, and the F-score is no longer higher than the baseline.

### Raw vs ICP-aligned viewer scores

Point-E and TripoSR predict in their own canonical pose. The `*_aligned` viewer runs rigidly rotate each prediction onto the GT with ICP (`nesegemaa/align.py`). The alignment **uses the GT**, so aligned scores are an **oracle upper bound**, not a fair result. They show that the priors lose mainly on pose, not shape (deck slide 13). Report raw scores as the result; label aligned ones as such.

## 3. PSNR + SSIM (rendering, multi-view references only)

These score a rendered held-out view against the real render, so they measure appearance, not 3D shape. They are never mixed with geometry scores.

- **PSNR** (dB, higher is better): from mean pixel error. +3 dB ≈ half the error.
- **SSIM** (0–1, higher is better): structural similarity.

NeRF oracle (`origin/ashry:results/oracle/summary.json`): **19.11 dB PSNR, SSIM 0.80** (0.804), on 24 objects at 128 px, 2,500 iterations, 20 train / 4 test views. Earlier versions of this file quoted 25.2 dB / 0.916: that was the first 6-object oracle (`451f58b:results/oracle/summary.json`, 2026-09-25), superseded by the 24-object run (`6b733b9`, later moved to `ashry` in `8a23409`).

## Comparability

Each track ran its own protocol (deck slide 9). Within a track, numbers are consistent. Across tracks, compare trends only.

| Track | Input at test | Test objects | Metric | Points | Aligned to GT |
|---|---|---|---|---|---|
| AtlasNet-SVR | 1 view | team split, 260 | squared Chamfer · F@τ=0.001 | 64,050 | no |
| Point-E, TripoSR | 1 view | team split, 260 | squared Chamfer · F@τ=0.001 | 4,096 | no |
| Pix2Vox, Pix2Vox++ | 3 views (00, 08, 16) | own seed-42 split, 263 (27 shared with the team test split) | IoU · CD-L1 · F@0.02 | 32³ voxels | no |
| COLMAP | 24 views + cameras | 294 (4 per class), 286 scored | CD-L1 · F@5% | ≈ 1,900 | by the known cameras |
| 3DGS | 20 views | all 1,689 completed of 1,694 | CD-L2 · F1@0.02 (= 5% radius) | 4,096 | scale + centre |
| NeRF oracle | 20 views | 24 | PSNR · SSIM | — | — |

The proposed fix (slide 20, pending team agreement): one split, the viewer scorer at 4,096 points with F@5%, one view at test, 3 seeds.

## Reading guide

| You see | Means |
|---|---|
| micro > macro | good on big classes, poor on rare ones |
| `noisy = True` | category has < 3 test objects; the number is noise |
| low Chamfer, low F | diffuse coverage, not a good shape: look at the outputs |
| viewer F@1% ≈ 0.27 | that is the ceiling, not a weak score |
| `*_aligned` | ICP onto GT: oracle, not a fair score |
| `best` ≫ `final` | overfitting or divergence; deploy the `best` epoch |
