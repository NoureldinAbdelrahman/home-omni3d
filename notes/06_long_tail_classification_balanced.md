# 06 — Long-tailed classification: our 31/69-class problem

## Papers

| Paper | Idea |
|---|---|
| Shen et al., **"Balanced Learning for Long-Tailed Recognition"** — survey, arXiv:2104.03705 | the field's survey of the exact problem |
| Cui et al., **"Class-Balanced Loss Based on Effective Number of Samples"** (CVPR 2019) — arXiv:1901.05555 | the effective-number weighting formula |
| Kang et al., **"Decoupling Representation and Classifier for Long-Tailed Recognition"** (ICLR 2020) — arXiv:1910.09217 | **most relevant: train representation on all data, re-sample only the classifier** |

## Background

The long tail is the reason our macro (per-category) mean is much lower than micro.

In our 69-category set:
- min = 2 objects (`plug`), max = 84 (`doll`), median = 20.
- 31 categories have **<3 test samples** — they are unlearnable, and even if they
  were just noise they drag down the average.
- If the loss is the *sum of per-sample losses*, a single big class contributes a
  large share of the gradient; the head classes are always over-represented。

Standard fixes, ordered cheapest to most invasive:

1. **Resample**: cap each class to N samples (`--max-per-class` in our
  `prepare_model_data.py`) — throw away data but balances the base distribution.
2. **Oversample rare classes**: repeat rare classes in the dataloader until
  per-class counts equal the mean count.
3. **Class-weighted loss**: weight each class by `w_c = 1 / n_c` (or
  `effective number of samples`: `(1 - beta) / (1 - beta^n_c)` from Cui et al.).
4. **Decouple (Kang et al.)**: train on all data as-is, then *retrain only the
  classifier head* on a class-balanced re-sampling at the end. Best empirical
  results for the least code change.

## Take-aways for our pipeline

- Our **loss** (BCE over voxels, Chamfer over points) is per-object, so there is
  *less* of a natural gradient imbalance in reconstruction than in classification,
  but the *eval* is unbalanced if we use a single mean.
- The main argument for a cap is *variance*, not accuracy: with rare classes the
  per-category IoU is noise. The paper answer: keep the **full-data numbers as the
  headline**, and show the per-category table separately (which the notebook does).
- If we want a single number that respects the should-not-train classes, the
  recommendation is: **exclude the `<3-object` categories from the headline**, or
  report both with and without.

## My notes

(add after reading)
