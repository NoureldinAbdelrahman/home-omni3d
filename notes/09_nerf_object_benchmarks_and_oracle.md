# 09 — NeRF object benchmarks & our multi-view oracle design

## Papers

| Paper | Idea |
|---|---|
| Mildenhall et al., **"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"** (ECCV 2020) — arXiv:2003.08934 | the NeRF formulation we reimplemented |
| Müller et al., **"Instant-NGP: Instant Neural Graphics Primitives"** (SIGGRAPH 2022) — arXiv:2201.00263 | multiresolution hash — why it's the "fast" oracle |
| Kerbl et al., **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (SIGGRAPH 2023) — arXiv:2308.14845 | the modern splatting oracle / required viewing |
| Yu et al., **"PixelNeRF: Neural Radiance Fields from One or Few Images"** (CVPR 2021) | few-image NeRF — the closest to what we'd like but *generalizes* |
| Barron et al., Mip-NeRF (ICCV 2021) | reduces NeRF artifacts via anti-aliasing |

## Background

Our `multiview_nerf_oracle.py` is a **pure-PyTorch** reimplementation of NeRF:
- a small MLP `F(x, d) -> (σ, rgb)` per object,
- volume-rendered from **20 training views**, evaluated on **4 held-out views**,
- PE L=10 for positions, L=4 for directions (higher than most papers use, fine).

It is deliberately **dependency-light** (no custom CUDA) because this machine's
`nvcc` is 12.8 and torch is `cu130`, so `gsplat`'s JIT kernels would hit an ABI
mismatch.

## Take-aways for our pipeline

- This is only *one* point on the multi-view spectrum, tuned for a 128-px render
  and 2500 iterations. If we want a better upper bound:
  1. **Instant-NGP's hash grid** would need a CUDA/torcher matching build —
     cannot compile here, skip.
  2. **Gaussian Splatting**: same GPU problem, but its splats can be exported to
     a point cloud *and* rendered, which would let us compare **geometry**
     (Chamfer/F-score) rather than just **PSNR** — that is the more useful oracle
     for a 3D course.
- **Metric alignment:** for a fair statement, our oracle should be reported with
  `PSNR / SSIM` *only* (radiance metrics), and not be mixed with IoU / Chamfer.
  The notebook keeps it on its own row for exactly that reason.

## My notes

(add after reading)
