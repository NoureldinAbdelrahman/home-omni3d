# house-omni3d

Bedroom/house subset of [OmniObject3D](https://opendatalab.com/OpenDataLab/OmniObject3D) bridged into **Pix2Vox** and **AtlasNet** for single-view 3D reconstruction (DMET 901).

Download 24-view Blender renders + point clouds for home categories (18 bedroom + 32 extra house), convert them into each model’s expected layout, and train / evaluate baselines on a GPU box.

## Contents

| Path | Role |
|------|------|
| `download_bedroom_subset.py` | Fetch renders + HDF5 point clouds from OpenDataLab |
| `download_house_extra.py` | Fetch **32 extra house categories** (kitchen / bath / utility) |
| `bedroom_categories.py` | Original 18-category subset + aliases (`lamp`→`light`, etc.) |
| `house_extra_categories.py` | Extra house category list (32) |
| `bedroom_omni_dataset.py` | Generic PyTorch loader over `dataset/` |
| `prepare_model_data.py` | Convert `dataset/` → Pix2Vox and AtlasNet layouts |
| `Pix2Vox/` | Vendored [Pix2Vox](https://github.com/hzxie/Pix2Vox) (voxel recon) |
| `AtlasNet/` | Vendored [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) (point / surface gen) |
| `dataset/` | **Data placeholder** (gitignored) |
| `requirements.txt`, `install_openxlab.sh` | Environment |

### Original bedroom categories (18)

`bed`, `pillow`, `chair`, `light`, `cabinet`, `table`, `sofa`, `stool`, `clock`, `tvstand`, `vase`, `tissue`, `teddy_bear`, `doll`, `plant`, `fan`, `suitcase`, `hair_dryer`

### Extra house categories (32)

Kitchen: `kettle`, `microwaveoven`, `ricecooker`, `pan`, `dish`, `cup`, `bowl`, `bottle`, `teapot`, `thermos`, `timer`  
Living/desk: `remote_control`, `speaker`, `projector`, `keyboard`, `laptop`, `monitor`, `mouse`, `power_strip`, `plug`  
Bathroom: `shampoo`, `soap`, `tooth_brush`, `tooth_paste`, `razor`, `medicine_bottle`  
Utility: `dustbin`, `fire_extinguisher`, `flash_light`, `hammer`, `scissor`, `umbrella`

## Setup

```bash
git clone https://github.com/NoureldinAbdelrahman/house-omni3d.git
cd house-omni3d

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./install_openxlab.sh          # openxlab without its broken setuptools pin
```

Optional OpenDataLab auth (or use `--use-fallback-keys`):

```bash
export OPENXLAB_AK=...
export OPENXLAB_SK=...
```

## Download data

```bash
# plan only (original 18 bedroom categories)
python download_bedroom_subset.py --dry-run

# full bedroom subset (~9 GB under dataset/)
python download_bedroom_subset.py --output-dir dataset --point-count 4096

# --- on a lab PC before the bedroom dump arrives ---
# list the 32 extra house categories
python download_house_extra.py --list

# plan
python download_house_extra.py --dry-run

# download extras into dataset/ (skips anything already local)
python download_house_extra.py --output-dir dataset --point-count 4096

# optional: auth if you have no cached login
python download_house_extra.py --use-fallback-keys
```

Layout written under `dataset/`:

```
dataset/
├── renders/<category>/<object_id>/{000..023}.png + transforms.json
└── point_clouds/<category>/<object_id>.npy
```

When you later copy the bedroom `dataset/` onto the same machine, both trees merge; re-run `prepare_model_data.py` so taxonomies cover every category on disk.

## Prepare model inputs

```bash
python prepare_model_data.py --target both
# or: --target pix2vox | --target atlasnet
# or pin a subset:
python prepare_model_data.py --categories kettle soap laptop
```

Categories are **auto-discovered** from `dataset/point_clouds/` and `dataset/renders/` (bedroom first, then sorted extras).

This fills (gitignored) placeholders:

- `Pix2Vox/datasets/OmniObject3D/` — taxonomy JSON, `00.png`–`23.png` renders, 32³ `.mat` voxels  
- `AtlasNet/dataset/data/` — `taxonomy.json`, `.npy` point clouds, render tree  

Symlinks point back into `dataset/`; copy the whole tree when moving machines, or re-run download + prepare.

## Train

### Pix2Vox (image → 32³ voxels)

```bash
cd Pix2Vox
python runner.py --epoch 50 --batch-size 8 --gpu 0
# eval a checkpoint
python runner.py --test --weights ./output/checkpoints/<run>/best-ckpt.pth --batch-size 8
```

### AtlasNet

Single-view (image-conditioned):

```bash
cd AtlasNet
python train.py \
  --class_choice bed chair cabinet light table doll clock pillow \
  --SVR --no_metro --nepoch 50 --batch_size 8 \
  --template_type SPHERE --dir_name log/svr_full --workers 2
```

Autoencoder (point cloud → surface):

```bash
python train.py \
  --class_choice bed chair cabinet light table doll clock pillow \
  --no_metro --nepoch 50 --batch_size 8 \
  --template_type SPHERE --dir_name log/ae_full --workers 2
```

Checkpoints land in `Pix2Vox/output/` and `AtlasNet/log/` (both gitignored).

## Moving machines / placeholders

Data is **not** in git. On a new machine:

1. Clone the repo.  
2. Either copy `dataset/` + prepared model dirs into the `.gitkeep` folders, **or** re-run download + `prepare_model_data.py`.  
3. Install deps and train.

Placeholder roots:

- `dataset/.gitkeep`  
- `Pix2Vox/datasets/OmniObject3D/.gitkeep`  
- `AtlasNet/dataset/data/.gitkeep`  

## Notes

- **Pix2Vox** patches: relative OmniObject paths in `config.py`, NumPy 2 / matplotlib / `torch.load(weights_only=False)` fixes, HWC tensorboard images.  
- **AtlasNet** patches: pure-Python `pymesh` shim, pure-PyTorch Chamfer fallback (no `nvcc`), PLY mesh reader, non-fatal HTML report.  
- Metrics (per milestone): IoU (Pix2Vox); Chamfer / F-score (AtlasNet); optional COV / MMD / FID.

## License

- Pipeline scripts: see repo.  
- Pix2Vox / AtlasNet: keep their upstream licenses under each subdirectory.

---

DMET 901 — 3D Object Generation from 2D Images.
