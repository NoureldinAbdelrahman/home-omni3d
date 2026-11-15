# Mohamed A. Awad (58-8189): Pix2Vox, Pix2Vox++

| | |
|---|---|
| Branch | `mohamed-ayman` (folder `mohamed_ayman/`, underscore) |
| Models | Pix2Vox, Pix2Vox++ |
| MS2 status | trained + evaluated |
| Protocol | 3 views at test (00, 08, 16), 224² input, own seed-42 split: 263 test objects (27 shared with the team test split); 32³ voxels; IoU@0.2–0.5, CD-L1, CD-L2, F@0.01/0.02 in the track's own frame |
| Results | `mohamed_ayman/outputs/evaluations/{pix2vox,pix2vox++}_3views_report.md` and `_test_results.csv`; training histories in `mohamed_ayman/outputs/checkpoints/*_3views_home_decor_history.json`; viewer runs `viewer/runs/mohamed_ayman__*` |
| Headline (micro) | IoU@0.4 0.333 / 0.327 · CD-L1 0.105 / 0.099 · F@0.02 0.236 / 0.256 · best val IoU 0.408 / 0.418 (Pix2Vox / Pix2Vox++) |
| Focus (slide 20, proposed) | keep Pix2Vox++, re-run with 1 view on the team split; drop Pix2Vox (≈ Pix2Vox++) |

Caveats: `configs/config.yaml` says 25 epochs, but the histories have 15 (Pix2Vox) and 30 (Pix2Vox++); the logs are authoritative. The viewer runs cover the 294 COLMAP objects, only 48 of them in this test split.

Code lives on `mohamed-ayman` (`Pix2Vox/`, `mohamed_ayman/src/`), not on `main`. Work only in `mohamed_ayman/` and `Pix2Vox/`. Full index: [`results/README.md`](../results/README.md).
