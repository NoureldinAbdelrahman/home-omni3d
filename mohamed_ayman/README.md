# Multi-View 3D Shape Reconstruction & Virtual Room Stager

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

## 📊 Benchmark Results

Evaluated on the held-out test split (263 3D objects, 6,312 multi-view images) across 76 categories with dedicated tracking on the **Home Decor Subset**:

| Model | Representation | Input Views | Val IoU / CD-L2 | Test Chamfer-L2 $\downarrow$ | F-Score@1% $\uparrow$ | Inference Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pix2Vox++** | $32^3$ Voxels | 3 views | **0.4078** (IoU) | 0.0184 | 34.2% | **18.4 ms** / obj |
| **AtlasNet** | 25 Surface Patches | 3 views | **0.0102** (CD) | **0.0142** | **41.7%** | **24.1 ms** / obj |
| **Point-E** | 4,096 Points | 1 view | 0.0245 (CD) | 0.0261 | 28.5% | 185.0 ms / obj |

### Key Metric Findings:
- **Multi-view advantage**: Moving from 1 view to 3 views resolves backside occlusion, increasing IoU by $+8.4\%$ and cutting Chamfer distance by $31\%$.
- **Macro vs. Micro Averaging**: Rare furniture classes (`bed`: 2 objects, `cabinet`: 7 objects) are weighted equally in Macro evaluation, preventing dominance by high-frequency novelty items like `doll` (84 objects).
- **Surface Quality**: AtlasNet excels at thin structures and sharp geometric corners (chair legs, table edges), while Pix2Vox++ ensures solid closed interiors.

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
