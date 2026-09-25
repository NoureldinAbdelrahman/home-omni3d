# 07 — Image-encoder pretraining & transfer: the AtlasNet encoder bug

## Papers

| Paper | Idea |
|---|---|
| ResNet paper (He et al. 2016) — arXiv:1512.03385 | why pretrained CNN features transfer |
| Huh et al., "What makes ImageNet good for transfer learning" (NeurIPS 2016) | empirical transfer study |
| Wu et al., **"Learning Single-image 3D Reconstruction by counting"** — see note 04, arXiv:1804.04210 | how SVR methods traditionally use CNN encoders |
| Tatarchenko et al. 2019, again — see note 05 | shows that not every SVR image encoder is pretrained |

## Background

Pix2Vox uses a **pretrained VGG16-BN** (ImageNet) image encoder → its encoder already
knows what an object looks like, and only has to learn the shape prior.

AtlasNet-SVR's image encoder in this repo is:
```python
resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
```
so the encoder is trained **from random initialization** on 1540 objects —
that is the whole image dataset. In published AtlasNet-SVR on full ShapeNet-13
(~35k models, ~2M Ave ) it converges despite a scratch encoder, but we cannot
replicate that with 1500 objects.

**This is a real bug/limitation in our AtlasNet baseline and can be fixed without
changing the supervision (still Chamfer).** The standard setup in the paper for SVR:

- Use a ResNet18/34 pretrained on ImageNet (or 3-color ImageNet), drop the final
  FC → replace with an MLP that outputs `bottleneck_size` (1024 here).
- Fine-tune the whole stack end-to-end.

## Take-aways for our pipeline

- **Action:** add a `--pretrained_encoder` flag to `AtlasNet` (or set it in
  `model/model.py`) and retrain. This is the most probable large jump for AtlasNet.
  - It keeps the model *image-conditioned* (the entire purpose of `--SVR`).
  - It is a fair fix when comparing with Pix2Vox (which uses a pretrained VGG16).

## My notes

(add after reading)
