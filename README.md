# home-omni3d

**Single-view 3D reconstruction of household objects.** DMET 901 Computer Vision, German University in Cairo. Team 9. Instructor: Dr. Mohamed Karam. TA: Eng. Rawan Darwish.

Task: given one RGB view (a 224² crop of a 1024² render), predict the full 3D shape as a 32³ voxel grid, a point cloud or a mesh. The data is a house-themed subset of [OmniObject3D](https://opendatalab.com/OpenDataLab/OmniObject3D) (licence CC BY 4.0; see [`docs/DATASET.md`](docs/DATASET.md#licence)).

**The MS2 deck is the reference for the project:** [`docs/ms2/ms2_problem_definition.pptx`](docs/ms2/ms2_problem_definition.pptx) (23 slides). Every claim in it is sourced in [`docs/ms2/ms2_outline.md`](docs/ms2/ms2_outline.md). Start with [`docs/README.md`](docs/README.md) for the doc map.

## Dataset scopes

These numbers describe different scopes and are all correct. Always say which one you mean.

| Scope | Categories | Objects | Images | Source |
|---|---:|---:|---:|---|
| OmniObject3D source (OpenDataLab) | 216 | 5,911 | 141,864 | [`docs/DATASET.md`](docs/DATASET.md) |
| Full download (all present objects complete) | 215 | 5,893 | 141,432 | [`docs/DATASET.md`](docs/DATASET.md) |
| Modelled subset, MS1 copy on disk (includes partial `laptop_002`) | 76 | 1,695 | 40,665 | [`results/ms1_snapshot.md`](results/ms1_snapshot.md) |
| **Modelled subset (MS2 benchmark)**, `laptop_002` dropped | **76** | **1,694** | 24 per object | `origin/ashry:ashry/benchmark/results/benchmark_v1.json` |
| Team split of the modelled subset (70/15/15, seeded) | 76 | 1,177 / 257 / 260 | | same file |

The 76 modelled categories = 18 bedroom (`bedroom_categories.py`) + 58 house extras. The 58 extras = 32 named in `house_extra_categories.py` + 26 more found on disk by `prepare_model_data.py`. Each object has 24 posed 1024² RGBA renders and a 4,096-point ground-truth cloud.

Long tail (benchmark file): doll has 84 objects; bed, canned_beverage and plug have 2 each; the median is 19.5; the top 10 classes hold 31.05% of the objects; 35 of 76 categories have fewer than 3 test objects.

## Branches, owners, models

`main` is the shared base: data pipeline, viewer and team docs. Each model has **one owner**. Its code and results live only on that owner's branch, under the owner's folder. Each track ran **its own protocol**, so the numbers are reported side by side and never ranked (see [`docs/METRICS.md`](docs/METRICS.md#comparability)).

| Member (ID) | Branch | Folder | Models | MS2 status |
|---|---|---|---|---|
| Mohamed A. Awad (58-8189) | `mohamed-ayman` | `mohamed_ayman/`, `Pix2Vox/` | Pix2Vox, Pix2Vox++ | trained + evaluated (3 views, own split, 263 test objects) |
| Noureldin S. Abdelrahman (58-7007) | `nesegemaa` | `nesegemaa/`, `AtlasNet/` | AtlasNet-SVR (trained); Point-E, TripoSR (frozen priors) | trained + evaluated / evaluated (frozen prior); team split, 260 test objects |
| Noureldeen A. Younes (58-1632) | `bones` | `bones/` | COLMAP (known poses, blind SfM) | scored, 286 of 294 objects; multi-view reference |
| Abdelrahman A. Ashry (58-0567) | `ashry` | `ashry/` | 3DGS, NeRF, NeuS | 3DGS scored (1,689 objects); NeRF oracle only; NeuS explored, no scores; multi-view references |
| Ahmed H. Mostafa (58-7370) | `hamdy` | `LGM/` | LGM | code only; dropped (team decision) |

Dropped at MS1→MS2: Pixel2Mesh and DeepSDF (were listed for `ashry`) and additive-subtractive modelling (was listed for `bones`, never implemented). TripoSR moved from `hamdy` to `nesegemaa`.

**Focus from here (slide 20, proposed, pending team agreement):** keep AtlasNet-SVR (SPHERE), Pix2Vox++ re-run with 1 view, and Point-E; keep COLMAP, 3DGS and NeRF as fixed references; drop Pix2Vox (≈ Pix2Vox++), TripoSR and LGM.

Headline numbers, each with its protocol and source file: [`results/README.md`](results/README.md).

Branch and folder names differ for one owner (`mohamed-ayman` branch, `mohamed_ayman/` folder). `nesegemaa` is a branch/folder name, not a person. Noureldin S. and Noureldeen A. are two different people. Naming rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Base layout (`main`)

| Path | Role |
|---|---|
| `download_bedroom_subset.py`, `download_house_extra.py`, `download_remaining.py` | Fetch data (modelled subset; remaining 140 categories) |
| `bedroom_categories.py`, `house_extra_categories.py` | The 18 + 32 *named* category lists (the other 26 modelled categories are auto-discovered) |
| `bedroom_omni_dataset.py`, `prepare_model_data.py` | Loaders, model-input conversion (drops incomplete objects such as `laptop_002`) |
| `check_dataset.py`, `fix_point_cloud_names.py` | Dataset integrity and point-cloud pairing |
| `viewer/` | Team 3D viewer and shared scorer ([`viewer/README.md`](viewer/README.md)) |
| `docs/` | Team docs ([`docs/README.md`](docs/README.md)) |
| `notes/` | Paper reading notes |
| `notebooks/dataset_analysis.ipynb` | MS1 dataset analysis and comparison figures |
| `results/` | MS2 result index, MS1 snapshot, MS1 figures |

Model code (`Pix2Vox/`, `AtlasNet/`, `LGM/`, owner folders) is **not** on `main`; see [`docs/ROADMAP.md`](docs/ROADMAP.md). MS1 snapshot: tag `milestone-1`.

## Setup

Requires Python 3.10–3.14 (not 3.14.1):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./install_openxlab.sh   # drops openxlab's broken setuptools pin; `pip check` must pass
```

## Data

```bash
python download_house_extra.py --include-bedroom --output-dir dataset --point-count 4096
python prepare_model_data.py --target both   # or: pix2vox | atlasnet
```

**Remaining categories** (the other 140 of the 216 on the remote, ~80 GiB): `download_remaining.py` fetches only the 24-view renders + 4096-point clouds. It is resumable (chunk journal + sha256 verification, atomic extraction) and prints live percent / speed / ETA. After any interruption, re-run the same command: finished categories are skipped and a partial archive resumes chunk by chunk.

**Full-download status: 215 of 216 categories, 5,893 of 5,911 objects (99.70%)**, every present object with all 24 views + its cloud. The 18 missing objects (category `package` blocked by the host's firewall, `laptop_002` and `anise_011` incomplete in the source) and per-category counts are in [`docs/DATASET.md`](docs/DATASET.md). Pairing is verified: `python fix_point_cloud_names.py --dry-run` (mapping now covers all categories) and `python check_dataset.py --dataset-dir dataset` must both come back clean on any copy of the dataset.

```bash
python download_remaining.py --output-dir dataset --dry-run   # plan + sizes, downloads nothing
python download_remaining.py --output-dir dataset             # Ctrl+C is safe; re-run to resume
```

Writes `dataset/renders/<cat>/<obj>/{000..023}.png` + `dataset/point_clouds/<cat>/<obj>.npy`. `dataset/` is gitignored: copy it between machines or re-download. Never commit `dataset/`, `*.pth`, `AtlasNet/log/` or `Pix2Vox/output/`.

> **Point-cloud pairing fix.** Datasets downloaded before this fix have ~95% of `point_clouds/<cat>/<obj>.npy` files holding a *different* object of the same category (the old downloader assumed the unnamed HDF5 rows were in sorted ID order; they are not). The shapes are intact, only the names are shuffled. Repair an existing copy offline, then rebuild model inputs and retrain/re-evaluate anything that used point clouds or voxels:
>
> ```bash
> python fix_point_cloud_names.py --dry-run   # report only
> python fix_point_cloud_names.py             # rename in place (idempotent)
> python prepare_model_data.py --target both
> ```
>
> New downloads use the named per-object PLY archive and are correct as downloaded.

## Results

- MS2 index (every track: owner, branch, path, metric, n, protocol, status): [`results/README.md`](results/README.md).
- Metric definitions, scales and comparability caveats: [`docs/METRICS.md`](docs/METRICS.md).
- Viewer runs (raw vs ICP-aligned, cited in MS2 or not): [`viewer/runs/README.md`](viewer/runs/README.md).
