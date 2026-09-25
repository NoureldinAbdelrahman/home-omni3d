# 08 — Modern generative priors for single-view 3D (optional stretch)

## Papers

| Paper | Idea |
|---|---|
| Achlioptas et al., **"Learning Representations and Generative Models for 3D Point Cloud Data"** (ICML 2018) — arXiv:1701.07749 | point cloud GAN/AE priors + COV/MMD metrics |
| Mittal et al., **"AutoSDF: Category-Aware Single-View 3D Shape Generation"** (CVPR 2022) — arXiv:2203.08070 | diffusion over a shape prior |
| Zhou et al., **"PVD: Point-Discrete Diffusion Models for 3D Point Cloud Generation"** (CVPR 2023-ish) — arXiv:2212.05991 | point-cloud diffusion |
| — optional: any triplane / transformer shape-GAN survey | stretch |

## Background

The last decade saw a transition from **discriminative regressors** (Pix2Vox,
AtlasNet) to **generative priors** on the shape side:

- Instead of predicting one shape, they sample a *plausible* shape from a
  distribution conditioned on the image. The image becomes a *prompt*, not an argmax.
- Diffusion / shape latent GANs work well when paired with dense GT, but need lots
  of data. For our long tail they may overfit.

## Take-aways for our pipeline

- This is the optional stretch for a "do better" pass. The **NeRF oracle** row
  gives us a reference; these methods would sit as *generative* baselines, not
  reconstructors, and will be much more expensive to train.
- First finish the cheap, fair wins (notes 05–07) before reaching for this.

## My notes

(add after reading)
