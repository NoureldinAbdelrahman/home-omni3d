# Contributing

## Branch ownership

- `main` is the shared base: data pipeline, viewer, team docs. It holds no model code and no model results.
- Each model has **exactly one owner**, and its results are reported only from that owner's branch:

| Branch | Owner | Folders the owner writes to |
|---|---|---|
| `mohamed-ayman` | Mohamed A. Awad | `mohamed_ayman/`, `Pix2Vox/`, `viewer/runs/mohamed_ayman__*` |
| `nesegemaa` | Noureldin S. Abdelrahman | `nesegemaa/`, `AtlasNet/`, `results/atlasnet/`, `viewer/runs/nesegemaa__*` |
| `bones` | Noureldeen A. Younes | `bones/`, `results/colmap/`, `viewer/runs/bones__*` |
| `ashry` | Abdelrahman A. Ashry | `ashry/`, `results/oracle/` |
| `hamdy` | Ahmed H. Mostafa | `LGM/`, `hamdy/` |

- Work only in your own folders. Changes to shared base files (`*.py` at the root, `viewer/`, `docs/`) go to `main` through a PR.
- Never rebase or force-push someone else's branch. Never rewrite history on `main`.
- Never commit `dataset/`, `.venv/`, checkpoints (`*.pth`, `*.pt`, `*.ckpt`), `AtlasNet/log/`, `Pix2Vox/output/` or `__pycache__/`. Small result files (`*.json`, `*.csv`, `*.md`, figures) and viewer predictions are committed.

## Result provenance

- A number in a doc or slide must point to a committed artifact (`<branch>:<path>`), not to a README table.
- Keep the artifact's value. If a doc rounds it, say so. If two sources disagree, keep the artifact value and record the conflict (for the deck: [`docs/ms2/ERRATA.md`](docs/ms2/ERRATA.md)).
- Always state the protocol next to a metric: views at test, test objects, point count, alignment, normalisation.
- One run = one directory. Do not overwrite or rename run directories: their names are cited.

## No leaderboard

Tracks used different inputs, splits, scorers and point counts ([`docs/METRICS.md`](docs/METRICS.md#comparability)). Put tracks side by side with their protocols; do not rank them or name a "best model" across tracks. Compare only within a track, on the same objects.

## Naming

Use these spellings in prose and tables. File and folder names stay as they are, because result paths cite them.

| Thing | Prose spelling | Spellings found in paths (keep) |
|---|---|---|
| People | Noureldin S. Abdelrahman, Noureldeen A. Younes, Mohamed A. Awad, Abdelrahman A. Ashry, Ahmed H. Mostafa | |
| Short forms | Noureldin S., Noureldeen A., Mohamed A., Abdelrahman A., Ahmed H. | |
| Branches | `mohamed-ayman`, `nesegemaa`, `bones`, `ashry`, `hamdy` (exact) | folder `mohamed_ayman/` (underscore) |
| AtlasNet-SVR | AtlasNet-SVR (AtlasNet when unambiguous) | `AtlasNet/`, `atlasnet` |
| Pix2Vox / Pix2Vox++ | Pix2Vox, Pix2Vox++ | `pix2vox`, `pix2vox++`, `pix2voxplusplus`, `Pix2VoxPlusPlus` |
| Point-E | Point-E | `point_e`, `pointe` |
| TripoSR | TripoSR | `triposr` |
| LGM | LGM | `LGM/` |
| COLMAP | COLMAP | `colmap` |
| 3DGS | 3DGS | `3dgs` |
| NeRF | NeRF | `nerf` |
| NeuS | NeuS | `neus` |
| Status words | trained + evaluated, evaluated (frozen prior), scored, reference, code only, dropped | |
| Metrics | IoU@0.4, CD-L1, CD-L2, F@0.02, F@5%, F@τ=0.001 (squared), PSNR, SSIM | |

`nesegemaa` is a branch and folder name, not a person: in prose write Noureldin S. "Noureldin" (`nesegemaa`) and "Noureldeen" (`bones`) are two people; do not merge the spellings.
