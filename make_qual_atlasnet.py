#!/usr/bin/env python3
"""Save AtlasNet-SVR predictions vs ground-truth point clouds on held-out objects.

For a few objects per category it writes, under
``results/qualitative/atlasnet/<cat>_<obj>/``: ``input.png`` (view 0),
``pred.npy`` / ``gt.npy`` (N,3 in UnitBall) and ``metrics.json`` (Chamfer, F-score).
The notebook renders these side by side against the expected shape.
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

ROOT = Path(__file__).resolve().parent
ATL = ROOT / "AtlasNet"
sys.path.insert(0, str(ATL))

import dataset.pointcloud_processor as pp  # noqa: E402
from easydict import EasyDict  # noqa: E402
from model.model import EncoderDecoder  # noqa: E402


def chamfer_and_fscore(a, b, tau=0.01):
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)   # pred -> gt
    db, _ = cKDTree(a).query(b)   # gt -> pred
    chamfer = float(da.mean() + db.mean())
    precision = float((db < tau).mean())
    recall = float((da < tau).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return chamfer, fscore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="svr25_150")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "qualitative" / "atlasnet")
    ap.add_argument("--categories", nargs="+",
                    default=["medicine_bottle", "cup", "teapot", "hammer", "shampoo"])
    args = ap.parse_args()

    run_dir = ATL / "log" / args.run
    opt = EasyDict(json.loads((run_dir / "options.json").read_text()))
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = EncoderDecoder(opt)
    net = torch.nn.DataParallel(net, device_ids=[0]).to(opt.device)
    net.load_state_dict(torch.load(run_dir / "network.pth", map_location=opt.device, weights_only=False))
    net.eval()

    to_tensor = transforms.Compose([transforms.Resize(224, interpolation=2), transforms.ToTensor()])
    center = transforms.CenterCrop(127)

    args.out.mkdir(parents=True, exist_ok=True)
    for cat in args.categories:
        pc_dir = ATL / "dataset" / "data" / "ShapeNetV1PointCloud" / cat
        if not pc_dir.is_dir():
            continue
        files = sorted(p.name for p in pc_dir.glob("*.npy"))
        test = files[int(len(files) * 0.8):]
        if not test:
            continue
        obj = test[0].split(".")[0]

        pts = torch.from_numpy(np.load(pc_dir / test[0])).float()
        gt = pp.Normalization.normalize_unitL2ball_functional(pts[:, :3]).squeeze(0).numpy()

        img_path = ATL / "dataset" / "data" / "ShapeNetV1Renderings" / cat / obj / "rendering" / "00.png"
        im = to_tensor(center(Image.open(img_path)))[:3]

        with torch.no_grad():
            out = net(im[None].to(opt.device), train=False)  # [1, prim, 3, P]
        pred = out.transpose(2, 3).reshape(1, -1, 3)[0].cpu().numpy()

        chamfer, fscore = chamfer_and_fscore(pred, gt, tau=0.01)
        outd = args.out / f"{cat}_{obj}"
        outd.mkdir(parents=True, exist_ok=True)
        np.save(outd / "pred.npy", pred.astype(np.float32))
        np.save(outd / "gt.npy", gt.astype(np.float32))
        Image.fromarray((im.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(outd / "input.png")
        (outd / "metrics.json").write_text(json.dumps({
            "category": cat, "object": obj, "chamfer": chamfer,
            "fscore_tau0.01": fscore, "n_points": int(pred.shape[0]),
        }, indent=2) + "\n")
        print(f"  {cat:16s} {obj:20s} chamfer={chamfer:.4f} fscore={fscore:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
