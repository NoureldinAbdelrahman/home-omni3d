# Results index (MS2)

Hand-maintained (not generated). It points to the committed artifact for every MS2 number. Values are copied from those files and rounded to 3 significant figures or less; the file is authoritative. Results live on each owner's branch: read them with `git show <branch>:<path>`.

The tracks used **different protocols** and are **not ranked** against each other ([`docs/METRICS.md`](../docs/METRICS.md#comparability)). Compare only within a row group.

Older material: [`ms1_snapshot.md`](ms1_snapshot.md) (MS1 notebook output) and [`comparison/`](comparison/README.md) (MS1 figures and table).

## Single-view tracks

| Track | Owner | Branch | Result path | Metric (micro) | n | Protocol | Status |
|---|---|---|---|---|---:|---|---|
| AtlasNet-SVR, SPHERE `a2_p25_sphere_s123` | Noureldin S. | `nesegemaa` | `results/atlasnet/a2_p25_sphere_s123/qual_agg.json` | Chamfer 0.0332 · F@τ=0.001 0.161 (macro 0.0348 / 0.165) | 260 | 1 view (00), team split, squared Chamfer, UnitBall, **64,050 pts**, not aligned | trained + evaluated |
| same, at matched density | Noureldin S. | `nesegemaa` | `nesegemaa/PAPER.md` §6.1 audit note | 4,096 pts: 0.0384 / 0.090 · 2,500 pts: 0.0409 / 0.073 | 260 | as above, prediction subsampled | trained + evaluated |
| AtlasNet-SVR, SPHERE seed 7 `a2_p25_sphere_s7` | Noureldin S. | `nesegemaa` | `nesegemaa/ablation_atlasnet.csv` (`qual_micro_ch`) | Chamfer 0.0343 · F 0.155 | 260 | as above | trained + evaluated |
| AtlasNet-SVR baseline `a0_svr25_imgnet` | Noureldin S. | `nesegemaa` | `results/atlasnet/a0_svr25_imgnet/qual_agg.json` | Chamfer 0.0467 · F 0.083 | 260 | 1 view, SQUARE, 25 patches, 2,500 pts | trained + evaluated |
| All 24 AtlasNet ablation runs | Noureldin S. | `nesegemaa` | `nesegemaa/ablation_atlasnet.csv`, `results/atlasnet/<run>/` | `qual_micro_ch`, `qual_micro_f` | 260 | see CSV columns (views, patches, template, points, lr, batch, seed) | trained + evaluated |
| TripoSR, masked | Noureldin S. | `nesegemaa` | `nesegemaa/ablation_triposr.csv` (tag `full`) | Chamfer 0.129 · F 0.089 (macro 0.124 / 0.090) | 260 | 1 view, masked, MC 128, 4,096 pts, not aligned | evaluated (frozen prior) |
| TripoSR, raw / crop | Noureldin S. | `nesegemaa` | same (`full_raw`, `full_crop`) | Chamfer 0.132 / 0.137 | 260 | as above | evaluated (frozen prior) |
| Point-E, default | Noureldin S. | `nesegemaa` | `nesegemaa/ablation_pointe.csv` (tag `full`) | Chamfer 0.186 · F 0.082 (macro 0.186 / 0.080) | 260 | 1 view, masked, base40M + upsampler, guidance 3, 4,096 pts | evaluated (frozen prior) |
| Point-E settings sweep | Noureldin S. | `nesegemaa` | `nesegemaa/ablation_pointe.csv` (other tags) | Chamfer moves ≤ 0.01 | **5** | 5-object panel only | evaluated (frozen prior) |
| Viewer rescoring of the three | Noureldin S. | `nesegemaa` | `viewer/runs/nesegemaa__*_full*/run.json`, `…a2_p25_sphere_s123/run.json` | see [viewer index](../viewer/runs/README.md) | 260 | viewer scorer, 4,096 pts, raw and ICP-aligned | evaluated |
| Pix2Vox | Mohamed A. | `mohamed-ayman` | `mohamed_ayman/outputs/evaluations/pix2vox_3views_report.md`, `pix2vox_3views_test_results.csv` | IoU@0.4 0.333 · CD-L1 0.105 · F@0.02 0.236 · best val IoU 0.408 | 263 | **3 views** (00, 08, 16), own seed-42 split (27 objects shared with the team test split), 32³ | trained + evaluated |
| Pix2Vox++ | Mohamed A. | `mohamed-ayman` | `mohamed_ayman/outputs/evaluations/pix2vox++_3views_report.md`, `pix2vox++_3views_test_results.csv` | IoU@0.4 0.327 · CD-L1 0.099 · F@0.02 0.256 · best val IoU 0.418 | 263 | as above | trained + evaluated |
| LGM | Ahmed H. | `hamdy` (use `origin/hamdy`) | `LGM/` (code at `ee0b1cc`, 2026-11-07) | **no scores committed** | — | — | code only, dropped |

Pix2Vox epochs: `mohamed_ayman/configs/config.yaml` says 25, but the committed histories (`mohamed_ayman/outputs/checkpoints/{pix2vox,pix2vox++}_3views_home_decor_history.json`) have 15 and 30 epochs. **The logs are authoritative; the config value is stale.** Best val IoU is the maximum `val_metric` in those histories. The config is not edited (owner decision).

Not reported (one owner per model): `mohamed-ayman:mohamed_ayman/outputs/evaluations/atlasnet_3views_*` (AtlasNet is owned by Noureldin S.).

## Multi-view references

| Track | Owner | Branch | Result path | Metric (micro) | n | Protocol | Status |
|---|---|---|---|---|---:|---|---|
| COLMAP, known poses | Noureldeen A. | `bones` | `results/colmap/sparse_known/summary.json` | CD-L1 0.134 · F@5% 0.539 · registered views 24.0 | 286 of 294 | 24 views + GT cameras, 4 objects per class, viewer scorer, ≈ 1,880 pts | scored, reference |
| COLMAP, blind SfM | Noureldeen A. | `bones` | `results/colmap/sparse_sfm/summary.json` | CD-L1 0.370 · F@5% 0.378 · registered views 14.3 | 286 of 294 | 24 views, no poses, Sim(3)-aligned to GT cameras | scored, reference |
| 3DGS | Abdelrahman A. | `ashry` (use `origin/ashry`) | `ashry/3dgs_metrics/benchmark_512_3k/benchmark_metrics.json` (`461f55c`) | F1@0.02 0.667 · F1@0.01 0.351 · CD-L2 mean 0.0035 (median 0.0011) | 1,689 of 1,694 | 20 views, 512 px, 3,000 iterations, 4,096 Gaussian centres, scale + centre aligned to GT bbox; 0.02 = 5% of the 0.4 GT radius | scored, reference |
| NeRF oracle | Abdelrahman A. | `ashry` (use `origin/ashry`) | `results/oracle/summary.json` | PSNR 19.11 dB · SSIM 0.80 (0.804) | 24 | 20 train / 4 held-out views, 128 px, 2,500 iterations | reference |
| NeRF, NeuS (full) | Abdelrahman A. | `ashry` | `ashry/nerf/`, `ashry/neus/` | **no scores committed** | — | — | explored |

COLMAP: 8 of 294 objects are excluded because the GT-to-camera fit put < 90% of points inside the silhouettes (`n_gt_unreliable` in the summary).

## Legacy and misnamed runs

All AtlasNet runs stay in `nesegemaa:results/atlasnet/`. Some names do not describe what ran:

| Run | What it actually is | Evidence |
|---|---|---|
| `a3_aug_rot`, `a3_aug_flip`, `a3_aug_aniso` | **Baseline rerun with a new random seed.** No augmentation was applied. | `AtlasNet/training/trainer.py:51` runs the augmenter only when `not opt.SVR`; each run's `options.json` has `SVR: true`, `manualSeed: null` |
| `a1_k3_attn` (and other `a1_k*`) | **Not single-view**: sees 3 (or 5, 8) photos at test | `ablation_atlasnet.csv` `K_views`; views 0, 8, 16 for K = 3 |
| `a2_p25_sphere_ft1e4` | Low-LR (1e-4) finetune of the SPHERE winner, 15 epochs: 0.0326 / 0.161, marginal | `ablation_atlasnet.csv` |
| `a0_svr25_warmdec` | Decoder warm-started from the official AtlasNet SVR-25 weights; no full-split score committed | `options.json` `reload_decoder_path` |
| `svr25`, `svr25_150`, `svr25_76`, `svr25_full`, `svr25_pretrained` | MS1 runs, trained before the point-cloud pairing fix (2026-10-09); also in `comparison/metrics_table.csv` | last changed 2026-10-04 |

Baseline settings were therefore trained four times: `a0_svr25_imgnet` 0.0467, `a3_aug_rot` 0.0424, `a3_aug_aniso` 0.0453, `a3_aug_flip` 0.0870 Chamfer (`qual_micro_ch`). The deck counts three (it does not flag `a3_aug_aniso`; see [`docs/ms2/ERRATA.md`](../docs/ms2/ERRATA.md)). Run-to-run noise is as large as most single-run gaps. Only SPHERE was repeated with seeds.

MS1 Pix2Vox runs (`results/pix2vox/<timestamp>/`) and the MS1 24-object NeRF oracle are on tag `milestone-1` (the first 6-object oracle is in `451f58b`); MS2 does not present them separately (the Pix2Vox track is Mohamed A.'s, the oracle is on `ashry`).
