# 04 — The "canonical template + deformation" family (AtlasNet's core idea)

**Group of papers that use a *canonical shape prior* + learned non-rigid deformation,
which is exactly what AtlasNet-SVR does**

Read in this order; the ideas build up.

| Paper | Idea |
|---|---|
| `Learning Shape Priors: T. Hernandez & 3D surface from shapes` (optional) | shape templates |
| Masci et al., `Geodesic Convolutional Neural Networks` (optional) | patch-based geometry |
| Groueix et al., **`AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation`** (CVPR 2018) — arXiv:1802.05384 | the paper behind our baseline |
| Kanazawa et al., **`Learning Single-Image 3D Reconstruction by Counting Relations`** (_, CVPR 2018, `CMR`) — arXiv:1804.04210 | differentiable mesh deformation from a single image; a good atlas-style alternative |

## Background (AtlasNet specifically)

```
image -> ResNet (from scratch in the reference implementation!)
      -> 1024-d latent z
      |
      `--- for each of K primitives (25 here):
              template patch (2D unit square)
              MLP mapping: patch + z -> 3D points
              |
      merge all primitives -> point cloud / mesh
      Training loss: Chamfer (only), no normals/normals
```

Key insight: the network does **not** need a shape prior per se. It owns a
fixed `K x N` 2D atlas that is deformed into 3D by the latent code. The
`generate_mesh` path is used only for evaluation and exports `.ply` files.
- K = 1 gives a coarse "cap"; K = 10–25 gives a detailed surface.
- Because the decoder is an MLP, it deforms points from **any** parameterization,
  so you can output a point cloud or a mesh from the same network.

Problems observed in practice (and the reason our score is low):
1. In eval mode, AtlasNet uses **regular points** on each patch — that's what our
   `make_qual_atlasnet.py` samples. It always *looks* like a nice surface, which can
   hide whether the underlying latent code is meaningful.
2. `--nb_primitives` is ours to choose. K=25 with 100 pts/primitive is
    "full shape" but is easy to underfit with MLPs; the original paper uses
    K=25 with `bottleneck 512` for the single-view model and gets its best
    **Chamfer ≈ 0.001** on full ShapeNet.
3. **The ImageNet ImageEncoder is `pretrained=False` in this repo.** Look at
  `AtlasNet/model/model.py`: `resnet.resnet18(pretrained=False, num_classes=...)`.
  In the paper the image encoder is a standard pretrained ResNet for the SVR
  experiments (see note 07 for why this matters).

## Take-aways for our pipeline

- Our AtlasNet number is dominated by an untrained image encoder. Switching the
  encoder to a pretrained ResNet18 → is the single biggest likely jump.
- Read note 05 first (F-score/Chamfer pitfalls) to make sure we are not fooling
  ourselves on the evaluation, and note 07 for the encoder fix.

## My notes

(add after reading)
