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

---

## Multi-View 3D Shape Reconstruction & Virtual Room Stager

**Course**: DMET 901 Computer Vision — German University in Cairo (GUC)  
**Instructor**: Dr. Mohamed Karam  
**Teaching Assistant**: Eng. Rawan  
**Author**: Mohamed Ayman Awad (ID: 58-8189, Tutorial 24)  
**Team**: Team 9 (`home-omni3d`)  

---

## 📌 Executive Summary

This subfolder houses **Mohamed Ayman's** complete multi-view 3D reconstruction pipeline, evaluation suite, and interactive Streamlit 3D application built on top of the **OmniObject3D** dataset.

Unlike single-view baselines which suffer from severe 2.5D visual hull ambiguity and occluded backside collapse, this framework leverages **multi-view fusion ($K \in \{1, 3, 5\}$ views)** to synthesize high-fidelity 3D geometry suitable for home decoration and virtual interior staging (chairs, beds, sofas, tables, cabinets, lamps, vases, pillows, etc.).

---

## 🏛️ Architecture & Implemented Models

### 1. Multi-View Pix2Vox++ (Volumetric Reconstruction)
- **2D Feature Encoder**: Pretrained ResNet-18 extracting compact spatial representations from each input view.
- **Context-Aware Multi-View Fusion**: Multi-channel scoring module computing view-wise spatial attention weights before collapsing across viewpoints.
- **3D Deconvolutional Decoder**: Transposed 3D convolutions scaling fused features into a dense $32 \times 32 \times 32$ voxel occupancy grid.
- **3D U-Net Refiner**: 3D residual encoder-decoder refining boundary voxels and smoothing thin furniture structures.
- **Compound Objective**: $\mathcal{L} = \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{coarse}}$.

### 2. Multi-View AtlasNet (Continuous Surface Deformation)
- **Deep Feature Backbone**: ResNet-50 extractor with multi-view symmetric pooling into a global latent vector $\mathbf{z} \in \mathbb{R}^{1024}$.
- **Parameterized Surface Patches**: 25 independent Multi-Layer Perceptrons (MLPs) deforming 2D square patches $[0, 1]^2$ into 3D continuous manifold surfaces (4,100 vertices total).
- **Surface Mesh Extraction**: Patch-wise 2D grid triangulation merged into a watertight 3D triangle mesh (`.obj` exportable).
- **Training Loss**: Vectorized Bidirectional Chamfer-L2 distance.

### 3. Point-E Baseline (Modern 3D Diffusion)
- OpenAI Point-E transformer baseline conditioned on RGB imagery with a morphological distance-transform visual hull depth prior fallback for offline/sandboxed deployment.

---

## 📊 Milestone 3 Benchmark Results (Full 215 Categories, 5,893 Objects)

Evaluated on the stratified held-out test split (**900 unseen objects** across **215 categories**, zero training leakage):

### 1. Best Model Per Category & Multi-View Generalization
| Category / Paradigm | Selected Best Model | Protocol | Voxel IoU@0.4 $\uparrow$ | CD-L2 $\downarrow$ | CD-L1 $\downarrow$ | F-Score@0.02 $\uparrow$ | Viewer F@5% $\uparrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Volumetric DL** | **Pix2Vox++ (MS3 Aug, ResNet-50 + Deep 3D U-Net)** | 3 views, General | **0.6446** | **0.0028** | **0.0566** | **0.4138** | **0.514** *(unseen)* / 0.537 |
| **Volumetric DL** | **Pix2Vox++ (MS3 Aug)** *(Single-View)* | 1 view, General | **0.6201** | **0.0031** | **0.0586** | **0.4063** | — |
| **Volumetric DL** | MS2 Baseline (`pix2vox`, Home Decor) | 3 views, Old | 0.3193 | 0.0123 | 0.1126 | 0.2311 | 0.302 |
| **Surface Mesh DL** | **AtlasNet (MS3 Aug, 25 Spherical Patches + Attention)** | 3 views, General | 0.5797 | **0.0021** | **0.0500** | **0.5040** | **0.581** *(unseen)* / 0.658 |
| **Surface Mesh DL** | **AtlasNet (MS3 Aug)** *(Single-View)* | 1 view, General | 0.5436 | 0.0026 | 0.0550 | 0.4632 | — |
| **Mathematics / SfM** | **COLMAP (`sparse_known` Reference)** | 24 views + GT Poses | — | — | 0.1341 *(viewer)* | **0.2242** *(F@2%)* | **0.539** *(pre-computed)* |

### 2. Data Augmentation Domain-Shift Ablation (300 Unseen Objects)
Comparing identical **Pix2Vox++ (ResNet-50 + Deep 3D U-Net + Class-Balanced)** trained with `--augment` vs `--no_augment`:

| Evaluation Domain | Without Augmentation (`--no_augment`) | **With 3D-Safe Augmentation (`--augment`)** | Domain Robustness Delta |
| :--- | :---: | :---: | :---: |
| **Clean White Background** (Std Test) | 0.6435 IoU | 0.6388 IoU *(0.4138 F@0.02)* | `-0.7%` (Comparable) |
| **Non-White / Cluttered Background** (Domain Shift) | **0.3372 IoU** *(-47.6% collapse)* | **0.6356 IoU** *(holds steady)* | **+88.5% Relative Gain** |

---

## 🖥️ Interactive Streamlit Web Application

The interactive web application (`app/app.py`) provides 4 dedicated modules:

1. **🛋️ 3D Reconstruction Studio**:
   - Select or upload real home furniture instances from OmniObject3D.
   - Choose input viewpoint count ($K \in \{1, 3, 5\}$).
   - Real-time 3D rotation, zoom, and inspection using Plotly WebGL.
   - Dynamic voxel threshold slider (0.20 – 0.70) to inspect thin leg structures vs surface noise.
   - Single-click **Download 3D Mesh (.obj)** for Blender / Unity.

2. **⚔️ Side-by-Side Model Arena**:
   - Direct tripartite comparison between **Pix2Vox++**, **AtlasNet**, and the **Physical Laser Scanner Ground Truth** on identical photographs.

3. **🏡 Virtual 3D Room Stager (Interior Design Mode)**:
   - Interactive 3D room canvas where reconstructed furniture can be positioned in $(X, Z)$ room coordinates.
   - Switchable floor materials (Hardwood Oak, Polished Marble, Minimalist Slate).

4. **📊 Dataset & Long-Tail Distribution Analytics**:
   - Visual distribution of physical object scales (60 mm to 2,800 mm) and class frequencies.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment
From this directory (`home-omni3d/mohamed_ayman`):
```bash
# Activate the project virtual environment
source ../../.venv/bin/activate

# Or install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

### 3. Evaluate Pretrained Models
```bash
# Evaluate Pix2Vox++ (3 views)
python src/evaluate.py --model pix2vox --weights outputs/checkpoints/pix2vox_3views_home_decor_best.pth --num_views 3

# Evaluate AtlasNet (3 views)
python src/evaluate.py --model atlasnet --weights outputs/checkpoints/atlasnet_3views_home_decor_best.pth --num_views 3
```

Results and Markdown benchmark reports will be written to `outputs/evaluations/`.

### 4. Train From Scratch
```bash
# Train Pix2Vox++ with 3 views on CUDA (AMP enabled)
python src/train.py --model pix2vox --num_views 3 --epochs 25 --batch_size 16

# Train AtlasNet with 3 views
python src/train.py --model atlasnet --num_views 3 --epochs 25 --batch_size 16
```

### 5. Run Automated Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 📁 Repository Structure

```
mohamed_ayman/
├── README.md               # This documentation
├── requirements.txt        # Python dependencies
├── configs/
│   └── config.yaml         # Hyperparameters & model configurations
├── app/
│   └── app.py              # Streamlit 3D Reconstruction Studio & Virtual Stager
├── src/
│   ├── data/
│   │   ├── dataset.py      # Multi-view loader with RGBA compositing & caching
│   │   ├── splits.py       # Stratified train/val/test splits (70/15/15)
│   │   └── voxelizer.py    # Vectorized point cloud voxelizer & Marching Cubes
│   ├── models/
│   │   ├── pix2vox.py      # Pix2Vox++ with Multi-View Fusion & 3D U-Net Refiner
│   │   ├── atlasnet.py     # Multi-View AtlasNet with 25 MLP surface patches
│   │   └── baseline_point_e.py # Point-E baseline with shape prior fallback
│   ├── losses/
│   │   ├── chamfer_distance.py # Chamfer Distance (L1/L2) & F-Score@tau
│   │   └── voxel_losses.py     # Soft Dice, Weighted BCE & Voxel IoU
│   ├── metrics/
│   │   └── evaluator.py    # Long-tail Macro & Micro benchmark aggregator
│   ├── train.py            # Unified training pipeline with mixed precision
│   └── evaluate.py         # Multi-model evaluation and reporting script
├── scripts/
│   └── prepare_milestone1_data.py # Exploratory analysis & EDA generation
├── tests/
│   └── test_pipeline.py    # Comprehensive test suite
└── outputs/
    ├── checkpoints/        # Saved model weights (.pth) & training logs (.json)
    ├── evaluations/        # Evaluation CSV sheets & Markdown summaries
    └── figures/            # EDA distribution & point cloud plots
```
