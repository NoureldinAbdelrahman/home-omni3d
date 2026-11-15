# Branch alignment checklist

Follow-up after `cleanup/ms2-consistency` is merged into `main`. **Docs-only** commits on each owner branch: no rebase, no force-push, no change to the branch's code or results. Each owner (or the presenter, with the owner's OK) applies their own branch.

Shared team docs to take from `main` on every branch:

```text
README.md  CONTRIBUTING.md
docs/README.md  docs/ROADMAP.md  docs/METRICS.md  docs/DATASET.md  docs/BRANCH_ALIGNMENT.md
docs/ms2/   (README.md, ERRATA.md, ms2_outline.md, ms2_problem_definition.pptx)
results/README.md  results/ms1_snapshot.md  results/comparison/README.md  notebooks/dataset_analysis.ipynb
viewer/runs/README.md  viewer/README.md
```

Generic step (per branch, after `git fetch`):

```bash
git switch <branch> && git merge --ff-only origin/<branch>
git checkout origin/main -- README.md CONTRIBUTING.md docs/README.md docs/ROADMAP.md docs/METRICS.md \
    docs/DATASET.md docs/BRANCH_ALIGNMENT.md docs/ms2 results/README.md results/ms1_snapshot.md \
    results/comparison/README.md notebooks/dataset_analysis.ipynb viewer/runs/README.md viewer/README.md
git status --short            # only docs paths may appear
git commit -m "docs: align team docs with main (MS2 consistency pass)"
git push origin <branch>      # normal push; open a PR if the owner prefers review
```

Do **not** blindly check out `<owner>/README.md` from `main` onto a branch whose owner has written a full README (see below). The `results/` and `viewer/` files apply only to branches that already carry them from the shared base (all except `hamdy`).

## `nesegemaa` (Noureldin S.)

| | |
|---|---|
| Copy from `main` | the shared list above; `nesegemaa/README.md`, `ashry/README.md`, `bones/README.md`, `hamdy/README.md`, `mohamed_ayman/README.md` (all stubs here) |
| Replace | `docs/ROADMAP.md` (its `nesegemaa`-only scope note says "AtlasNet + Point-E"; the new ROADMAP covers TripoSR and all branches); `docs/ms2/ms2_outline.md` (rewritten on `main`) |
| Needs owner decision | `ashry/notes.md`, `bones/notes.md`, `hamdy/notes.md`, `mohamed_ayman/notes.md` on this branch are stale stubs for other owners (Pixel2Mesh/DeepSDF, additive-subtractive, TripoSR under Hamdy). Suggest deleting them or replacing each with a one-line pointer to that owner's README |
| Preserve | `nesegemaa/notes.md` (run log), `nesegemaa/PAPER.md`, `nesegemaa/*.csv`, `results/atlasnet/`, `results/qualitative/`, `AtlasNet/`, `viewer/runs/nesegemaa__*` |
| Note | `PAPER.md` §6.5 still says "rotation mild win; flips hurt" for runs that never augmented, and §6.1 says "~11%" where the numbers give 12.4% (see `docs/ms2/ERRATA.md`). Owner's call whether to edit |

## `mohamed-ayman` (Mohamed A.)

| | |
|---|---|
| Copy from `main` | the shared list above; `ashry/README.md`, `hamdy/README.md`, `nesegemaa/README.md` (stubs here) |
| Keep, add the card | `mohamed_ayman/README.md` is the owner's full README: keep it, and add the owner-card table from `main:mohamed_ayman/README.md` at the top. Keep `bones/README.md` as is (it is the bones COLMAP README, merged from `bones`) |
| Preserve | `Pix2Vox/`, `mohamed_ayman/src/`, `mohamed_ayman/outputs/`, `viewer/runs/mohamed_ayman__*` |
| Owner decision | `mohamed_ayman/configs/config.yaml` says 25 epochs; histories have 15 / 30. Either fix the config or add a comment that the logs are authoritative. `outputs/evaluations/atlasnet_3views_*` is outside this track (AtlasNet is owned by Noureldin S.): keep or remove at the owner's discretion; it is not reported |

## `bones` (Noureldeen A.)

| | |
|---|---|
| Copy from `main` | the shared list above; `ashry/README.md`, `hamdy/README.md`, `mohamed_ayman/README.md`, `nesegemaa/README.md` |
| Keep, edit | `bones/README.md` is the full COLMAP README: keep it, add the owner-card table from `main:bones/README.md` at the top, and change the title "Bones - COLMAP, Additive-Subtractive Modeling" to drop additive-subtractive (dropped, never implemented) |
| Preserve | `bones/*.py`, `results/colmap/`, `viewer/runs/bones__*` |

## `ashry` (Abdelrahman A.)

Local `ashry` is behind `origin/ashry`; work from `origin/ashry`.

| | |
|---|---|
| Copy from `main` | the shared list above; `bones/README.md`, `hamdy/README.md`, `mohamed_ayman/README.md`, `nesegemaa/README.md` |
| Keep, add the card | `ashry/README.md` is the owner's full README: keep it, add the owner-card table from `main:ashry/README.md` at the top |
| Preserve | `ashry/{3dgs,nerf,neus,benchmark,app,3dgs_metrics}/`, `results/oracle/` |
| Owner decision | `ashry/benchmark/results/benchmark_v1.json` checksums predate the point-cloud pairing fix (ERRATA E7): re-freeze. NeRF / NeuS scores: commit them if they exist |

## `hamdy` (Ahmed H.)

Local `hamdy` is behind `origin/hamdy`; work from `origin/hamdy`. This branch forked before the viewer and results index existed on `main` (merge-base `3d5efbc`), so `viewer/` and `results/comparison/` are absent here.

| | |
|---|---|
| Copy from `main` | `CONTRIBUTING.md`, `docs/` files in the shared list, `hamdy/README.md` (owner card: LGM, code only, dropped; replaces "placeholder — currently unassigned"). Skip the `viewer/` and `results/` files unless the owner merges `origin/main` first (a merge, not a rebase) |
| Owner decision | the root `README.md` here differs heavily from `main`: check what it holds before replacing it |
| Preserve | `LGM/` |

## After all branches

Re-run on each branch:

```bash
git grep -n -E 'Pixel2Mesh|DeepSDF|additive-subtractive|25\.2|0\.916|18 \+ 32' -- . ':!docs/ms2/ms2_outline.md' ':!docs/archive'
```

Hits should be historical or scoped only (the owner READMEs and `docs/ms2/ERRATA.md` mention dropped models on purpose).
