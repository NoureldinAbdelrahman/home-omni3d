# Cleanup report: MS2 consistency pass (2026-11-15)

Branch `cleanup/ms2-consistency` (worktree `../OmniObject-cleanup`), based on `origin/main` @ `879a350` (local `main` was one commit behind). Not pushed, no PR. Docs-only: no training, evaluation or notebook re-run, no result value changed, no owner branch touched, the `.pptx` not edited (blob `a5c2eb4`, identical to `nesegemaa`).

Commits:

| Commit | Content |
|---|---|
| `docs: import docs/ms2 from nesegemaa; cleanup plan` | `git checkout nesegemaa -- docs/ms2` + plan |
| `docs: align README, ROADMAP, METRICS and DATASET with MS2` | README, CONTRIBUTING (new), docs/README (new), ROADMAP, METRICS, DATASET |
| `results: add MS2 index and label the MS1 snapshot` | results/README (MS2 index), results/ms1_snapshot.md, results/comparison/README, notebook output path |
| `viewer: add run index across branches` | viewer/runs/README, link from viewer/README |
| `docs: replace owner stubs with consistent owner cards` | 5 owner READMEs |
| `docs: rewrite MS2 outline for the 23-slide deck; add ERRATA and ms2 README` | docs/ms2/* |
| `docs: add per-branch alignment checklist` | docs/BRANCH_ALIGNMENT.md |
| `docs: final cleanup report` | this file |

## Deck vs repo disagreements, and what was kept

Full list with sources: [`docs/ms2/ERRATA.md`](docs/ms2/ERRATA.md).

| Where | Disagreement | Kept |
|---|---|---|
| Slide 5 | "bed, plug: 2" omits canned_beverage; "median ≈ 20" vs 19.5 | artifact: three categories, 19.5 |
| Slides 11, 15, 16 (notes) | baseline settings "trained three times"; `a3_aug_aniso` also never augmented → four runs (0.0424, 0.0453, 0.0467, 0.0870) | artifact: four runs (new finding, verified in `trainer.py:51` + `options.json`) |
| Slide 21 (notes) | "LGM next to be run and evaluated" vs on-slide "Dropped" | deck on-slide: dropped |
| Slide 21 (notes) | "Phase 2 is where every branch is now" vs Phase 2 "done" | on-slide phase bar |
| Slide 4 (notes) | 58 extras "auto-discovered"; really 32 named + 26 auto-discovered | 32 + 26 |
| Slide 4 (notes) | split "frozen with SHA-256 checksums": manifest hashes predate the pairing fix | split valid, checksums stale |
| Slides 19, 22 | 3DGS "F@5%" is `f1_at_0.02` from a different scorer with bbox alignment | label F1@0.02 (≈ F@5%), different scorer |
| Slide 9 | COLMAP "4 per class" (76 × 4 ≠ 294) | "up to 4 per class" |
| Slide 6 | LGM in MS2 scope; superseded by slide 20 | dropped |
| `docs/METRICS.md` | NeRF oracle 25.2 dB / 0.916 | 19.11 dB / 0.804 (25.2 was the 6-object run in `451f58b`) |
| `docs/METRICS.md` | AtlasNet "F@1% (τ = 0.01)", "~2,500 points" | τ² = 0.001 squared (≈ 3.2% radius); SPHERE 64,050 pts with matched 4,096 / 2,500 numbers |
| README branch table | ashry = Pixel2Mesh/DeepSDF/3DGS/Nvidia NeRF; hamdy = TripoSR; nesegemaa = Point-E/AtlasNet; bones = COLMAP/additive-subtractive | deck slide 21 |
| `docs/ROADMAP.md` | "single source of truth", results on every branch, push to `origin/main` | rewritten branch-aware |
| `PAPER.md` §6.1 (`nesegemaa`) | "~11%" gain at matched 2,500 pts; the numbers give 12.4% (deck: −12%) | deck; PAPER not edited |
| `mohamed_ayman/configs/config.yaml` | 25 epochs vs histories 15 / 30 | logs; config not edited |

Every headline number in the canonical table of the task was re-checked against its artifact (AtlasNet `qual_agg.json` + ablation CSVs, Pix2Vox reports/CSVs/histories, COLMAP summaries, 3DGS `benchmark_metrics.json`, oracle `summary.json`, `benchmark_v1.json` long-tail and split counts, category files 18 / 32 / 26). All match within the deck's rounding.

## Scopes clarified

| Number | Scope | Where now stated |
|---|---|---|
| 5,911 / 216 | OmniObject3D source | README, DATASET |
| 5,893 / 215 / 141,432 images | full download, complete objects | README, DATASET |
| 1,695 / 40,665 images | MS1 on-disk copy of the 76 categories, includes partial `laptop_002` | README, DATASET, `results/ms1_snapshot.md` banner |
| 1,694 | MS2 benchmark (76 categories, `laptop_002` dropped) | README, DATASET, owner cards |
| 1,177 / 257 / 260 | team split (seed 2026) | README, DATASET, METRICS |
| 263 (27 shared) / 294 (286) / 1,689 / 24 | Pix2Vox split / COLMAP / 3DGS / NeRF oracle test sets | METRICS comparability table, results index |
| "~1,700" (old README) | removed | |

## Deleted, moved, renamed

| Item | Action | Reason |
|---|---|---|
| `project_guidelines.pdf` (untracked, root of the `nesegemaa` checkout) | deleted | byte-identical (`cmp`) to tracked `docs/project_guidelines.pdf` |
| `ms2_problem_definition (1).pptx` (untracked, root) | moved to `../OmniObject-local-archive/` | the colleague's older 16-slide copy (2026-11-07); not a source, not committed |
| `results/README.md` → `results/ms1_snapshot.md` | `git mv` + hand-added banner | MS1 generated output, labelled; the generated text below the banner is unedited |
| `notebooks/dataset_analysis.ipynb` | output path `results/README.md` → `results/notebook_index.md` (2 source lines) | so a re-run cannot overwrite the MS2 index or the snapshot. Stored cell outputs unchanged |

No tracked file deleted (`git diff --name-status origin/main | grep ^D` is empty). `Pix2Vox/` (untracked, 1.4 GB on `nesegemaa`) left as is: its code lives on `mohamed-ayman`. `results/comparison/metrics_table.csv` kept under its name (the notebook writes it and the outline cites it) and labelled MS1 in `results/comparison/README.md`.

## Decisions taken

- No merged MS2 comparison table: the tracks are not comparable, so `results/README.md` lists each track with its own protocol, grouped single-view / multi-view.
- Owner cards live in `<owner>/README.md` on `main`. The `notes.md` stubs exist only on `nesegemaa` and are left for that branch (BRANCH_ALIGNMENT).
- Outline: current 23-slide source map and open questions on top; all earlier entries kept verbatim under a dated History section (headings demoted one level).
- Cross-branch citations use `origin/ashry` and `origin/hamdy` (local copies are behind).

## Needs owner decision

1. `nesegemaa`: delete or replace the stale `ashry/notes.md`, `bones/notes.md`, `hamdy/notes.md`, `mohamed_ayman/notes.md` stubs; whether to correct `PAPER.md` §6.5 ("rotation mild win; flips hurt" for runs that never augmented) and §6.1 ("~11%").
2. `mohamed-ayman`: fix `config.yaml` epochs; keep or remove `outputs/evaluations/atlasnet_3views_*` (outside this track).
3. `ashry`: re-freeze `benchmark_v1.json` checksums on the fixed clouds; commit NeRF / NeuS scores if they exist.
4. `hamdy`: root `README.md` differs heavily from `main`; branch forked before the viewer.
5. Team: slide 20 focus and single protocol; improvement proposals for Mohamed A., Noureldeen A., Abdelrahman A.; confirm Ahmed H. Mostafa = `hamdy`.
6. Whether to merge `cleanup/ms2-consistency` into `main` (not pushed).

## Owner branches still needing a docs-only update

All five: `nesegemaa`, `mohamed-ayman`, `bones`, `ashry`, `hamdy`. Steps and paths to preserve: [`docs/BRANCH_ALIGNMENT.md`](docs/BRANCH_ALIGNMENT.md).

## Verification run

| Check | Result |
|---|---|
| `git status --short` (worktree) | clean after commits |
| `git status --short` (`nesegemaa` checkout) | only `?? Pix2Vox/` |
| `git diff --stat origin/main` | 23 files, +837 / −276 lines; only binary is the imported deck (6.6 MB, already on `nesegemaa`) |
| `git diff --check origin/main` | clean |
| `git diff --name-status origin/main \| grep ^D` | none |
| relative Markdown links + anchors in all tracked `.md` | 0 broken |
| `git cat-file -e` on 30 cited cross-branch paths and 11 cited commits | all exist |
| stale-term grep (`Pixel2Mesh\|DeepSDF\|additive-subtractive\|25\.2\|0\.916\|1,?695\|18 \+ 32`) | only scoped / historical hits: dropped-model notes, the scope tables, the ERRATA, MS1 notebook outputs, the MS1 snapshot, a literature mention in `notes/03_*.md`, hash substrings in `fix_point_cloud_names.json` |
| notebook JSON | still valid after the path edit |
| `viewer/view.py` with `viewer/runs/README.md` present | loads `runs/*/run.json` only, so the README is ignored |
