# Abdelrahman A. Ashry (58-0567): 3DGS, NeRF, NeuS

| | |
|---|---|
| Branch | `ashry` (local copies may be behind: use `origin/ashry`) |
| Models | 3DGS, NeRF, NeuS (per-object, multi-view) |
| MS2 status | 3DGS scored (1,689 objects); NeRF oracle only; NeuS explored, no scores; multi-view references |
| Protocol | views 000–019 in, 020–023 held out (`ashry/benchmark/results/benchmark_v1.json`, which also defines the team split). 3DGS: 20 views, 512 px, 3,000 iterations, 4,096 Gaussian centres, scale + centre aligned to the GT bbox, GT radius 0.4. NeRF oracle: 24 objects, 128 px, 2,500 iterations, 20 train / 4 test views |
| Results | `ashry/3dgs_metrics/benchmark_512_3k/benchmark_metrics.{json,csv}` (`461f55c`); `results/oracle/summary.json` |
| Headline | 3DGS F1@0.02 0.667 (= F@5% of radius), F1@0.01 0.351, CD-L2 mean 0.0035; NeRF oracle 19.11 dB PSNR, SSIM 0.80 |
| Focus (slide 20, proposed) | 3DGS and NeRF stay as fixed references, no new runs |

Dropped: Pixel2Mesh and DeepSDF (listed here at MS1).

Code lives on `ashry` (`ashry/`). Work only in `ashry/` and `results/oracle/`. Full index: [`results/README.md`](../results/README.md).
