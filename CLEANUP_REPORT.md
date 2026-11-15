# Cleanup report: MS2 consistency pass

Status: **plan** (written before editing; the final report replaces this section).

Branch `cleanup/ms2-consistency`, worktree `../OmniObject-cleanup`, based on `origin/main` @ `879a350`.

## Will change
- Team-facing docs on `main`: `README.md`, `CONTRIBUTING.md` (new), `docs/README.md` (new), `docs/ROADMAP.md`, `docs/METRICS.md`, `docs/DATASET.md`.
- Import `docs/ms2/` from `nesegemaa`, rewrite `ms2_outline.md` for the 23-slide deck, add `docs/ms2/README.md` and `ERRATA.md`.
- `results/README.md` becomes a hand-maintained MS2 index; the MS1 notebook output moves to a labelled snapshot.
- Indexes: `results/comparison/README.md`, `viewer/runs/README.md`.
- Owner cards in `<owner>/README.md`, plus `docs/BRANCH_ALIGNMENT.md`.
- Remove the untracked root duplicates in the `nesegemaa` working tree (not tracked, so they are not commits).

## Will not touch
- The `.pptx`, owner branches, result JSON/CSV/figures, viewer predictions, checkpoints, `dataset/`, `Pix2Vox/`, `AtlasNet/`.
- No training, evaluation or notebook re-run.
