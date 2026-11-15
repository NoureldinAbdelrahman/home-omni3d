# MS2 deck errata

The deck (`ms2_problem_definition.pptx`, 23 slides) is not edited. These are the places where it, its speaker notes or `ms2_outline.md` differ from the committed artifacts. In every case the repo docs keep the **artifact value**.

| # | Where | Deck says | Artifact / repo says | Kept | Source |
|---|---|---|---|---|---|
| E1 | Slide 5 (on-slide) | "bed, plug: 2" | bed, canned_beverage and plug have 2 objects each (the speaker notes list all three) | three categories | `origin/ashry:ashry/benchmark/results/benchmark_v1.json` |
| E2 | Slide 5 (on-slide) | "median ≈ 20" | median 19.5 | 19.5 (deck value is a rounding) | same file |
| E3 | Slides 11, 15, 16 (notes; slide 15/16 "same settings: 0.042–0.087") | baseline settings trained **three** times (`a0`, `a3_aug_rot`, `a3_aug_flip`) | **four**: `a3_aug_aniso` (0.0453) also never augmented (`SVR: true`, augmenter skipped at `AtlasNet/training/trainer.py:51`). The 0.042–0.087 range is unchanged | four runs: 0.0424, 0.0453, 0.0467, 0.0870 | `nesegemaa:nesegemaa/ablation_atlasnet.csv`, `nesegemaa:results/atlasnet/a3_aug_aniso/options.json` |
| E4 | Slide 21 (notes) | "Ahmed H.'s LGM pipeline code is in; it is next to be run and evaluated" | on-slide status and slide 20: LGM **dropped** (team decision, 2026-11-15) | dropped, code only | slides 20–22 |
| E5 | Slide 21 (notes) | "Phase 2 is where every branch is now" | the on-slide phase bar says Phase 2 done, Phase 3 in progress | on-slide phase bar | slide 21 |
| E6 | Slide 4 (notes) | "58 extra house categories, auto-discovered by prepare_model_data.py" | 32 of the 58 are named in `house_extra_categories.py`; 26 are auto-discovered | 32 named + 26 auto-discovered | `house_extra_categories.py`, `benchmark_v1.json` |
| E7 | Slide 4 (notes) | split "frozen with SHA-256 checksums in ashry's benchmark manifest" | the manifest (2026-10-06) holds **pre-pairing-fix** hashes: 449 of 470 checked local clouds hash to a different name. Evaluation loads by name, so scores use the fixed clouds, but the checksums do not verify the current files | the split lists are valid; the checksums are stale | `ms2_outline.md` feedback pass (2026-11-15) |
| E8 | Slides 19, 22 | 3DGS "F@5% 0.667" next to COLMAP "F@5% 0.539" | 3DGS is `f1_at_0.02` on a 0.4-radius GT (= 5% of the radius) from Abdelrahman A.'s own scorer, with bbox scale + centre alignment; COLMAP is the team viewer scorer. Same tolerance, different scorer and alignment | label as F1@0.02 (≈ F@5%), different scorer | `origin/ashry:ashry/3dgs_metrics/benchmark_512_3k/benchmark_metrics.json`, `bones:results/colmap/sparse_known/summary.json` |
| E9 | Slide 9 | COLMAP "294, 4 per class" | 76 × 4 = 304: classes with fewer than 4 objects contribute fewer | "up to 4 per class, 294 objects" | `bones:results/colmap/sparse_known/summary.json` |
| E10 | Slide 6 (on-slide) | MS2 committed scope lists LGM as a frozen prior | superseded by slide 20 (LGM dropped). Not an error at the time of writing; read slide 6 as history | dropped | slides 6, 20 |

Repo-side corrections found during the same pass (not deck errors, already fixed in the docs):

- `docs/METRICS.md` quoted the NeRF oracle as 25.2 dB / 0.916 SSIM: that was the first 6-object run (`451f58b`). The current oracle is 19.11 dB / 0.804 on 24 objects (`origin/ashry:results/oracle/summary.json`).
- `docs/METRICS.md` called the AtlasNet F-score "F@1% (τ = 0.01)". The code uses τ² = 0.001 on squared distances (≈ 3.2% of the radius), which is what the deck calls F@τ=0.001.
- `nesegemaa/PAPER.md` §6.1 says the matched-2,500-point Chamfer gain is "~11%"; from its own numbers (0.0409 vs 0.0467) it is 12.4%, as on slide 11. `PAPER.md` is on `nesegemaa` and is not edited here.
- `mohamed_ayman/configs/config.yaml` lists 25 epochs; the committed histories have 15 / 30 (the deck uses the logs). Owner to fix the config.
