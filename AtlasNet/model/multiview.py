import torch
import torch.nn as nn

import model.resnet as resnet


class MultiViewEncoder(nn.Module):
    """Shared ResNet-18 + symmetric pooling over K views -> single code.

    Mirrors the multi-view pattern of ``mohamed-ayman/src/models/atlasnet.py``
    (per-view features, max-pool to a global latent), adapted to our
    bottleneck-sized codes and our ImageNet-pretrained encoder:

    - the SAME ``resnet.resnet18(pretrained=True, num_classes=...)`` backbone
      as the single-view path (same init, same provenance assertion);
    - pooling is permutation-invariant (views are unordered);
    - ``K=1`` reduces exactly to the single-view path (max/softmax over one
      element is the identity) — asserted numerically before any training.

    Input is expected ImageNet-normalized already (``EncoderDecoder`` applies
    the normalization buffers before dispatching here).
    """

    def __init__(self, opt, pool="max"):
        super(MultiViewEncoder, self).__init__()
        assert pool in ("max", "attn"), f"unknown pool {pool!r}"
        self.pool = pool
        self.encoder = resnet.resnet18(pretrained=True,
                                       num_classes=opt.bottleneck_size)
        if pool == "attn":
            self.attn = nn.Linear(opt.bottleneck_size, 1)

    def forward(self, x):
        # x: [B, K, 3, H, W]; a lone [B, 3, H, W] is treated as K=1.
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, K = x.shape[:2]
        z = self.encoder(x.reshape(B * K, *x.shape[2:])).reshape(B, K, -1)
        if self.pool == "max" or K == 1:
            return z.max(dim=1).values
        w = torch.softmax(self.attn(z).squeeze(-1), dim=1)  # [B, K]
        return (w.unsqueeze(-1) * z).sum(dim=1)
