# Metrics guide — how to read our results

Plain-language reference for every number the notebook reports.
See `docs/ROADMAP.md` for the pipeline itself.

---

## 1. IoU at thresholds t = 0.20 / 0.30 / 0.40 / 0.50 — Pix2Vox (voxel model)

The model outputs, for each of the 32³ = 32,768 little cubes, a score 0–1 meaning
*"I think this cube is inside the object."* To get a solid shape you pick a
**threshold t**: score ≥ t → solid, else empty.

```
IoU = (cubes solid in BOTH prediction and ground truth)
      ────────────────────────────────────────────────
      (cubes solid in EITHER prediction or ground truth)
```

- **1.0 = perfect overlap, 0 = no overlap at all.** There is no universal
  "good/bad" line — for this task ~0.3 is mediocre; published numbers reach 0.6–0.7.
- **Why 4 thresholds?** A model can cheat by predicting a *bigger* shape
  (helps at low t) or a *smaller* one (helps at high t). Reporting all four stops
  either trick. Our best is at t ≈ 0.2–0.3 and we are weak at t=0.5 → our model
  predicts **too big / diffuse** shapes (visible in the notebook: red fringes
  around the teal expected shape).
- **micro vs macro** (per-category report):
  - *micro* = one big average over all test objects → big classes dominate.
  - *macro* = average of per-category averages → every category counts equally,
    exposing that tiny categories (bed, fork) score near zero.

## 2. Chamfer + F-score@τ — AtlasNet (point-cloud model)

AtlasNet outputs ~2,500 predicted points; the expected shape has 4,096 real points.
Shapes are normalized so the object fits a unit sphere, so distances below are in
"object-radius units".

**Chamfer** (lower = better, 0 = perfect): every predicted point → nearest real
point → average those distances, *and* the reverse direction, then add the two.
It is an **average mismatch distance**: `chamfer = 0.04` ≈ error of 4% of the
object's radius.

Problem: it is an average, so a lazy model that dumps a tiny blob in the middle
can still get a so-so score.

**F-score@τ** (we use τ = 0.01, i.e. 1% of the object radius): stricter, and
catches those blobs.

- *Precision* = of my predicted points, how many are within 0.01 of a real point?
  (I'm not inventing fake surface)
- *Recall* = of the real points, how many are covered by a predicted point?
  (I covered the whole object)
- **F = 2·P·R/(P+R)** — 1 = perfect, 0 = nothing within tolerance.

Example from our results: `cup` has chamfer 5.84 **and** F ≈ 0.0006 — the blob's
average distance looks "meh" but essentially **zero** points land within 1% — a
failure Chamfer alone hides. The tables therefore always show both.

**best vs final**: *best* = best epoch on validation (the model we deploy);
*final* = last epoch. If `final << best`, the model was overfitting — for the
early AtlasNet runs the final validation loss was 100–1000× worse than best.

## 3. PSNR + SSIM — NeRF oracle (rendering, not geometry)

The oracle reconstructs *appearance*: it renders a held-out camera view and
compares pixel-by-pixel to the real photo.

- **PSNR** (dB, higher = better): built from average pixel error. 20 dB = clearly
  wrong, 30+ dB = looks close; +3 dB ≈ halves the error, +10 dB = 10× less error.
  Ours: **25.2 dB** average.
- **SSIM** (0–1, higher = better): perceptual similarity of structure/contrast.
  Ours: **0.916**.

**It is not comparable to IoU/Chamfer** — it measures "how good is a *view*"
(20 input views, photometric), not "how good is the *3D shape*" (1 input view,
geometric). The notebook keeps it on its own row; it is the ceiling multi-view
data can reach, not a rival single-view score.

## 4. Reading guide for the notebook

| You see | Means |
|---|---|
| IoU 0.33 @ t=0.3 | about a third of the shape is right; far from 0.6+ published |
| micro > macro | model does well on big classes, poorly on rare ones |
| `noisy = True` | that category has <3 test objects — the number is essentially noise |
| F-score 0.08 | almost nothing within 1% tolerance → AtlasNet is still very weak |
| NeRF 25 dB | decent reconstruction *if given 20 views* — the gap single-view must close |
| `best` ≫ `final` | overfitting; deploy the `best` epoch |
