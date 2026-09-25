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

## Update — verified bug #2 in Pix2Vox, and both fixes now run

While auditing the checkpoints I found that **Pix2Vox never used its pretrained
weights either**, in a sneakier way than AtlasNet:

1. `Pix2Vox/models/encoder.py` loads `vgg16_bn(pretrained=True)` ✅
2. but `core/train.py` then called `encoder.apply(init_weights)`, which
   **re-initializes every Conv/BN in the encoder — including the frozen VGG**;
3. those params are excluded from the optimizer (`requires_grad=False`), so the
   VGG stayed a **frozen random** feature extractor for all 250 epochs.

Evidence from the old `best-ckpt.pth`: VGG conv std 0.2728 ≈ kaiming theory
0.2722 (pretrained: 0.1807); BN γ exactly 1.0 (pretrained varies); max diff
vs ImageNet weights 1.71.

**Fixes implemented (both smoke-tested, ImageNet tensors verified byte-exact
after training):**

| model | change |
|---|---|
| Pix2Vox | `core/train.py` initializes only `encoder.layer1..3`, never `encoder.vgg` |
| AtlasNet | `model/resnet.py` loads ImageNet weights (fc head rebuilt to bottleneck); `model/model.py` uses `pretrained=True` + ImageNet input normalization as registered buffers (saved in the checkpoint) |

**Results (69 categories, same data/schedule as the from-scratch runs):**

| metric | from scratch | pretrained | Δ |
|---|---|---|---|
| Pix2Vox IoU t=0.5 (test) | 0.1432 | **0.2079** | +45% |
| Pix2Vox IoU t=0.3 (test) | 0.2954 | **0.3122** | +5.7% |
| Pix2Vox macro t=0.5 | 0.1365 | **0.1932** | +42% |
| AtlasNet best F-score | 0.0817 | 0.0829 | +1.5% |
| AtlasNet best Chamfer | 0.0452 | **0.0405** | −10% |
| AtlasNet final val loss | 15.3 (exploded) | **0.043 (stable)** | ~stable |

Conclusion: pretraining was the single biggest Pix2Vox win (it also fixes the
"predicts too fat shapes" failure — IoU gain concentrates at strict thresholds).
For AtlasNet the encoder fix mainly **stabilized training** (val loss no longer
explodes; per-object chamfer outliers like cup 5.84 → 0.21), but F-score is
still ~0.08 — the remaining AtlasNet gap is data scale / long tail (see note 06),
not initialization. See `docs/METRICS.md` for how to read these numbers.

## My notes

(add after reading)
