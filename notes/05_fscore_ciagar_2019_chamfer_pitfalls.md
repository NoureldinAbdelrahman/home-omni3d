# 05 — Chamfer & F-Score pitfalls: how to read our errors fairly

## Papers for this note

| Paper | What it clarifies |
|---|---|
| Tatarchenko et al., **"What Do Single-View 3D Reconstruction Networks Learn?"** (CVPR 2019) — arXiv:1905.03678 | the huge measurement / ranking pitfalls in this exact field |
| Van der Maaten et al., `Deep metric learning and Chamfer distance` — not a paper, but the background for τ |  |
| Chen, Buyik, K. Xu & G. Zhang, `AUnified Model: Point cloud ... [ion metric/F-score](#)` (CGAL `F-Score@tau`) | F-score definition |

## Background

**Two evaluation numbers, two failure modes.**

- **Chamfer Distance (CD)**: average (or sum) of nearest-neighbor distances
  in both directions. It is very sensitive to ****.
- **F-Score@τ** = precision/recall on the two Chamfer distances, evaluated at a
  **τ** distance tolerance. Above τ, an object is "reconstructed"; below it, a
  point is considered isolated / wrong.

### Why this matters for our numbers

1. **We bin the target cloud to a unit sphere**. Published numbers often use
  a *bounding-box- or unit-ball-normalized* target. Our `make_qual_atlasnet.py`
  uses `unitL2Ball`, matching AtlasNet's training normalization — so the comparison
  is internally consistent, but not directly comparable to published numbers unless
  they use the same normalization.
2. **A tiny blob collapses to an easy chamfer mean.** A prediction where every point
  sits at 0.5 of the expected surface radius has CD ~ 0.5, which "looks reasonable".
   The **F@τ=0.01** we use is the **right** metric for detecting that (it collapses
   to 0.0006 for cup, versus ~0.015 for hammer), and it is exactly what detects the
  cup/shampoo collapse in our figures.
3. **F-score val is an average over all val samples.** A single catastrophic sample
  dominates it; the paper corrects this by reporting per-category or the
  **macro** average. Our notebook already prints both.
4. Tatarchenko's paper is the field's most important caveat: many SVR networks do
  not learn true correspondence with the input image — they mostly *memorize
  category centroids* and add a bit of shape-specific variation. It proposes
  `chamfer`-based bench-tests to decide whether the model actually uses the image.

## Take-aways for our pipeline

- To claim "the model is using the image", do Tatarchenko's **conditioned vs
  unconditioned** test: train a variant that ignores the image (`--no_learning` is
  not the same; instead replace the image input with a zero vector) and compare its
  Chamfer/F-score to the real model. If the numbers are close, the model has not
  learned a conditioned prior.
- Always report **both** CD and F@τ in our comparison table (we do).
- τ=0.01 in UnitBall space is a *reasonable* τ; τ=1% of diameter. Keep it fixed.

## My notes

(add after reading)
