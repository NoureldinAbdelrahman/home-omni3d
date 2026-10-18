#!/usr/bin/env python3
"""Tatarchenko-style input-dependence probe on the POST-FIX AtlasNet winner.

The MS1 probe (`results/ablation/image_usage.json`) ran on a pre-fix
checkpoint trained on shuffled image->cloud pairs, where ignoring the image
was the optimal strategy — doubly suspect. This re-runs the test on
`a2_p25_sphere_s123` (best-model weights, corrected data, full 260-object
test split, view 0):

  real:     the object's own view-0 image (same preprocessing as atlas_qual)
  constant: uniform mid-gray image (no instance information at all)
  shuffled: another test object's view-0 image (seed-0 derangement)

If the winner reconstructs the instance it sees: real << shuffled ~= constant.
If it is a category prior: real ~= shuffled ~= constant (MS1 pattern).

Usage: python nesegemaa/atlas_input_ablation.py [--limit N]
Writes nesegemaa/atlas_input_ablation_s123.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
ATL = ROOT / "AtlasNet"
sys.path.insert(0, str(ATL))
sys.path.insert(0, str(ROOT))

import dataset.pointcloud_processor as pp  # noqa: E402
from easydict import EasyDict  # noqa: E402
from model.model import EncoderDecoder  # noqa: E402

TAU_SQ = 0.001
RUN = "a2_p25_sphere_s123"


def chamfer_and_fscore(a, b, tau_sq=TAU_SQ):
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    da2, db2 = da ** 2, db ** 2
    chamfer = float(da2.mean() + db2.mean())
    precision = float((db2 < tau_sq).mean())
    recall = float((da2 < tau_sq).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return chamfer, fscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_dir = ATL / "log" / RUN
    wpath = run_dir / "best-model.pth"
    assert wpath.is_file(), f"{wpath} missing"
    opt = EasyDict(json.loads((run_dir / "options.json").read_text()))
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = EncoderDecoder(opt)
    net = torch.nn.DataParallel(net, device_ids=[0]).to(opt.device)
    net.load_state_dict(torch.load(wpath, map_location=opt.device, weights_only=False))
    net.eval()
    print(f"loaded best-model.pth from {run_dir} on {opt.device}", flush=True)

    splits = json.loads((ATL / "dataset" / "data" / "splits.json").read_text())
    pairs = [(c, o) for c, sp in sorted(splits.items()) for o in sp.get("test", [])]
    if args.limit:
        pairs = pairs[:args.limit]

    to_tensor = transforms.Compose([transforms.Resize(224, interpolation=2), transforms.ToTensor()])
    center = transforms.CenterCrop(127)

    def load_im(cat, obj, view=0):
        p = ATL / "dataset" / "data" / "ShapeNetV1Renderings" / cat / obj / "rendering" / f"{view:02d}.png"
        return to_tensor(center(Image.open(p)))[:3]

    def load_gt(cat, obj):
        pts = torch.from_numpy(np.load(
            ATL / "dataset" / "data" / "ShapeNetV1PointCloud" / cat / f"{obj}.npy")).float()
        return pp.Normalization.normalize_unitL2ball_functional(pts[:, :3]).squeeze(0).numpy()

    def predict(im):
        with torch.no_grad():
            out = net(im[None].to(opt.device), train=False)
        return out.transpose(2, 3).reshape(1, -1, 3)[0].cpu().numpy()

    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    # derangement: no object keeps its own image
    while any(o == i for i, o in enumerate(order)):
        order = rng.permutation(len(pairs))
    real_ims = [load_im(c, o) for c, o in pairs]
    const_im = torch.full_like(real_ims[0], 0.5)

    agg = {k: {"ch": [], "f": []} for k in ("real", "constant", "shuffled")}
    for i, (cat, obj) in enumerate(pairs):
        gt = load_gt(cat, obj)
        ims = {"real": real_ims[i], "constant": const_im,
               "shuffled": real_ims[int(order[i])]}
        for k, im in ims.items():
            ch, fs = chamfer_and_fscore(predict(im), gt)
            agg[k]["ch"].append(ch)
            agg[k]["f"].append(fs)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)}", flush=True)

    out = {"run": RUN, "weights": "best-model.pth", "n": len(pairs),
           "tau_squared": TAU_SQ, "split": "splits.json test lists, view 0"}
    for k in ("real", "constant", "shuffled"):
        out[k] = {"micro_chamfer": float(np.mean(agg[k]["ch"])),
                  "micro_fscore": float(np.mean(agg[k]["f"]))}
    (ROOT / "nesegemaa" / "atlas_input_ablation_s123.json").write_text(
        json.dumps(out, indent=2) + "\n")
    for k in ("real", "constant", "shuffled"):
        print(f"{k:9s} chamfer={out[k]['micro_chamfer']:.4f} fscore={out[k]['micro_fscore']:.4f}")
    print("wrote nesegemaa/atlas_input_ablation_s123.json")


if __name__ == "__main__":
    main()
