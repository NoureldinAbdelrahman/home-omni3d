# ROADMAP — house-omni3d (DMET 901: 3D Object Generation from 2D Images)

This document is the **single source of truth** for an agent working on the lab PC.
It explains the pipeline, where every artifact belongs, and exactly where to put
results. Read it fully before touching anything.

---

## 1. Project summary

OmniObject3D **house-themed subset** bridged into two single-view reconstruction
baselines:

| Model | Task | Output | Metric |
|-------|------|--------|--------|
| **Pix2Vox** | image → 32³ voxel grid | `.mat` / voxel volume | IoU per threshold |
| **AtlasNet (`--SVR`)** | image → surface | `.ply` mesh / point cloud | Chamfer, F-score |

Course milestone (DMET 901, GUC) asks for image-conditioned 3D generation plus
quantitative comparison. Optional secondary metrics: COV / MMD / FID.

### Category sets
- **Bedroom (18)**: bed, pillow, chair, light, cabinet, table, sofa, stool,
  clock, tvstand, vase, tissue, teddy_bear, doll, plant, fan, suitcase, hair_dryer
- **Extra house (32)**: kettle, microwaveoven, ricecooker, pan, dish, cup, bowl,
  bottle, teapot, thermos, timer, remote_control, speaker, projector, keyboard,
  laptop, monitor, mouse, power_strip, plug, shampoo, soap, tooth_brush,
  tooth_paste, razor, medicine_bottle, dustbin, fire_extinguisher, flash_light,
  hammer, scissor, umbrella

Categories on disk are **auto-discovered** by `prepare_model_data.py`; never
hardcode them in model code.

---

## 2. Environment / reproduction

```bash
git clone https://github.com/NoureldinAbdelrahman/home-omni3d.git
cd home-omni3d
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10–3.14 (not 3.14.1)
pip install -r requirements.txt
./install_openxlab.sh
pip check          # must print: No broken requirements found.
```

### Data
```bash
# extras only (~11.7 GB)
python download_house_extra.py --output-dir dataset --point-count 4096

# everything (18 bedroom + 32 extras, ~21 GB)
python download_house_extra.py --include-bedroom --output-dir dataset --point-count 4096

# build model inputs from whatever is in dataset/
python prepare_model_data.py --target both
```

`dataset/` layout:
```
dataset/
├── renders/<category>/<object_id>/{000..023}.png + transforms.json
└── point_clouds/<category>/<object_id>.npy        # (N,3) float32
```

If the 9 GB bedroom dump is copied in later, **re-run `prepare_model_data.py`**
so taxonomies include every category.

---

## 3. Training

### Pix2Vox — image → 32³ voxels
```bash
cd Pix2Vox
python runner.py --epoch 50 --batch-size 8 --gpu 0
python runner.py --test --weights ./output/checkpoints/<run>/best-ckpt.pth --batch-size 8
```
`--gpu 0` = first GPU (sets `CUDA_VISIBLE_DEVICES`). Checkpoints →
`Pix2Vox/output/checkpoints/<timestamp>/`.

### AtlasNet `--SVR` — image → surface (this is the image-conditioned one)
`--class_choice` **must** be passed (default is `airplane`). Build it from the
taxonomy:
```bash
cd AtlasNet
CLASSES=$(python -c "import json;print(' '.join(d['synsetId'] for d in json.load(open('dataset/data/taxonomy.json'))))")
python train.py --class_choice $CLASSES --SVR --no_metro \
  --nepoch 50 --batch_size 8 --nb_primitives 25 --template_type SQUARE \
  --dir_name log/svr25 --workers 2
```
- `--nb_primitives 1` = a single surface patch (coarse). Use 10–25 for a full shape.
- Autoencoder (point-cloud→surface) is an *extra* baseline, not image-conditioned:
  omit `--SVR`.
- Checkpoints → `AtlasNet/log/<dir_name>/network.pth`.

Both are long — run under `tmux`.

---

## 4. Where results go (IMPORTANT)

Create these directories and place artifacts **exactly** here. Use the
`<timestamp>` / `<dir_name>` produced by the runs; do not invent names.

```
results/
├── README.md                      # index table (created by the notebook)
├── pix2vox/
│   └── <run-id>/
│       ├── stdout.log             # full console output (incl. TEST RESULTS table)
│       ├── config.txt             # `pprint(cfg)` dump from runner
│       ├── test_results.csv       # parsed per-category IoU table
│       └── summary.json           # {overall_per_t, best_t, best_iou, n_samples}
├── atlasnet/
│   └── <dir_name>/
│       ├── log.txt                # json_stats per epoch
│       ├── options.json
│       ├── curve.png / curve_log.png
│       └── summary.json           # chamfer/fscore best+final
└── comparison/
    ├── metrics_table.csv          # one row per model with headline metrics
    └── figures/                   # exported PNGs from the notebook
```

### Rules
- **Never commit raw data, checkpoints (`.pth`), or `dataset/`** — all gitignored.
- `results/**/stdout.log`, `*.csv`, `*.json`, `*.png` **are** committed (small).
- One run = one timestamped dir. Don't overwrite; re-runs get new dirs.
- Pix2Vox `TEST RESULTS` table format (tab-separated):
  `Taxonomy  #Sample  Baseline  t=0.20  t=0.30  t=0.40  t=0.50` then
  `Overall  <best t column>`. Parse defensively — sample counts of 1 are noise.

---

## 5. Notebook: `notebooks/dataset_analysis.ipynb`

The notebook must run top-to-bottom on a machine with `dataset/` present. It:

1. **Discover** categories & counts from `dataset/point_clouds` and `dataset/renders`.
2. **Dataset info**: per-category object/img counts, long-tail table + bar chart,
   split sizes from taxonomy JSONs, class balance.
3. **Integrity / failure points**:
   - missing renders (`< 24` PNGs, missing `transforms.json`)
   - missing point clouds, wrong shape (`(N,3)`), NaNs/infs, duplicate points
   - point-cloud extent/anisotropy, degenerate (flat) clouds
   - objects present in one modality only
4. **Cleaning / imputation suggestions** (printed as an actionable checklist):
   - drop or flag objects with <24 views / missing PC
   - skip/store degenerate clouds; prefer re-download for NaN clouds
   - class-imbalance handling: cap per-class, oversample rare, or use
     class-weighted loss; note categories with <3 objects are not trainable
5. **Results ingestion**: read `results/**/summary.json` + CSVs if present and
   render comparison tables/plots; if absent, print "no runs found yet".
6. **Export**: write `results/README.md` index and `results/comparison/figures/*.png`.

Keep cells defensive (missing files → informative message, not a crash).

---

## 6. Definition of done for the agent

- [ ] `results/` tree populated per §4 with the real run(s) from this machine.
- [ ] `notebooks/dataset_analysis.ipynb` runs end-to-end with no exceptions.
- [ ] `results/README.md` + `results/comparison/metrics_table.csv` regenerated.
- [ ] Nothing large/ignored accidentally staged (`git status` review).
- [ ] Commit + push to `origin/main`.
