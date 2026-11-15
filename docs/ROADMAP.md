# Roadmap: pipeline and where things live

This guide covers the pipeline and where each artifact lives. It does not define project scope or status: those come from the MS2 deck ([`docs/README.md`](README.md#source-of-truth-order)).

## 1. What `main` does

`main` is the shared base. It holds no model code and no model results (they were moved to owner branches in `3d5efbc`; the MS1 state with everything on one branch is tag `milestone-1`).

| Step | Command | Output |
|---|---|---|
| Environment | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && ./install_openxlab.sh && pip check` | `.venv/` (gitignored) |
| Modelled subset (76 categories) | `python download_house_extra.py --include-bedroom --output-dir dataset --point-count 4096` | `dataset/renders/<cat>/<obj>/{000..023}.png + transforms.json`, `dataset/point_clouds/<cat>/<obj>.npy` (4,096 × 3) |
| Remaining 140 categories (optional) | `python download_remaining.py --output-dir dataset` | same layout ([`DATASET.md`](DATASET.md)) |
| Integrity / pairing | `python check_dataset.py --dataset-dir dataset`; `python fix_point_cloud_names.py --dry-run` | both must be clean |
| Model inputs | `python prepare_model_data.py --target both` | `Pix2Vox/datasets/OmniObject3D/*`, `AtlasNet/dataset/data/*` (gitignored); drops incomplete objects (`laptop_002`) |
| Analysis | `notebooks/dataset_analysis.ipynb` | MS1 figures in `results/comparison/figures/`, `results/comparison/metrics_table.csv` and `results/notebook_index.md` |
| Viewer | `python viewer/view.py` | local page; runs read from `viewer/runs/` |

Categories are auto-discovered from `dataset/` by `prepare_model_data.py`: the 18 bedroom + 32 named extras in the category files, plus 26 more on disk, give the 76 modelled categories. Never hardcode categories in model code.

The prepared model inputs go to `Pix2Vox/` and `AtlasNet/` paths. Those directories exist only on the owner branches that carry the code (below). On `main` the prep script creates the data folders only.

## 2. Where each owner's code and results live

Read across branches without checking them out: `git show <branch>:<path>`, `git cat-file -e <branch>:<path>`. Prefer `origin/<branch>`: local copies may be stale.

| Track | Owner | Branch | Code | Results |
|---|---|---|---|---|
| Pix2Vox, Pix2Vox++ | Mohamed A. Awad | `mohamed-ayman` | `Pix2Vox/`, `mohamed_ayman/src/`, `mohamed_ayman/configs/config.yaml` | `mohamed_ayman/outputs/evaluations/*_3views_{report.md,test_results.csv}`, `mohamed_ayman/outputs/checkpoints/*_history.json`, `viewer/runs/mohamed_ayman__*` |
| AtlasNet-SVR, Point-E, TripoSR | Noureldin S. Abdelrahman | `nesegemaa` | `AtlasNet/`, `nesegemaa/` | `results/atlasnet/<run>/`, `nesegemaa/ablation_{atlasnet,pointe,triposr}.csv`, `nesegemaa/PAPER.md`, `viewer/runs/nesegemaa__*` |
| COLMAP | Noureldeen A. Younes | `bones` | `bones/` | `results/colmap/sparse_{known,sfm}/`, `viewer/runs/bones__*` (also copied to `main`) |
| 3DGS, NeRF, NeuS | Abdelrahman A. Ashry | `ashry` | `ashry/{3dgs,nerf,neus,benchmark,app}/` | `ashry/3dgs_metrics/benchmark_512_3k/`, `results/oracle/summary.json`, `ashry/benchmark/results/benchmark_v1.json` (team split) |
| LGM | Ahmed H. Mostafa | `hamdy` | `LGM/` (`infer.py`) | none (code only, dropped) |

The full index of result files with protocols is [`results/README.md`](../results/README.md).

## 3. Reproducing a track

Check out the owner's branch (read-only unless you are the owner) and follow its own README:

| Track | Entry point |
|---|---|
| Pix2Vox / Pix2Vox++ | `mohamed-ayman:mohamed_ayman/README.md` |
| AtlasNet-SVR, Point-E, TripoSR | `nesegemaa:nesegemaa/PAPER.md` §4 (setup), `nesegemaa:nesegemaa/notes.md` (run log) |
| COLMAP | `bones:bones/README.md` |
| 3DGS / NeRF / NeuS | `origin/ashry:ashry/README.md` |
| LGM | `origin/hamdy:LGM/readme.md` |

For AtlasNet-SVR, `--class_choice` must be passed (the default is `airplane`). Build it from the taxonomy written by `prepare_model_data.py`:

```bash
cd AtlasNet   # on nesegemaa
CLASSES=$(python -c "import json;print(' '.join(d['synsetId'] for d in json.load(open('dataset/data/taxonomy.json'))))")
python train.py --class_choice $CLASSES --SVR --no_metro --nepoch 50 --batch_size 8 \
  --nb_primitives 25 --template_type SPHERE --dir_name log/<run> --workers 2
```

In SVR mode AtlasNet's point-cloud augmentation flags have no effect (`AtlasNet/training/trainer.py:51`): see the run-name caveat in [`results/README.md`](../results/README.md#legacy-and-misnamed-runs).

## 4. Rules for results

- Results are committed on the owner's branch, not on `main`. `main` only carries indexes, team docs and the viewer runs that were promoted with the viewer.
- Never commit `dataset/`, checkpoints (`*.pth`), `AtlasNet/log/`, `Pix2Vox/output/`. Small JSON/CSV/MD/PNG results and viewer predictions are committed.
- One run = one directory. Do not overwrite or rename: the deck and the outline cite run names.
- Every result states its protocol: views at test, split and number of test objects, scorer and point count, alignment. See [`METRICS.md`](METRICS.md).
- To show predictions in the team viewer, export from your own evaluation code with `viewer/export.py` ([`viewer/README.md`](../viewer/README.md)).

## 5. Phases (deck slide 21)

| Phase | State at MS2 |
|---|---|
| 1. Data pipeline (download, prep, 70/15/15 split, integrity checks) | done, on `main` |
| 2. Run models (each owner's branch) | done |
| 3. Evaluate (each track with its own evaluator, protocol recorded) | in progress |
| 4. Improve (MS3; tracks side by side in the viewer, no cross-track ranking) | next |

The slide 20 proposal (one protocol for everyone: team split, viewer scorer at 4,096 points, F@5%, one view at test, 3 seeds; the bigger 5,893-object dataset; a pose fix for the priors) is pending team agreement.
