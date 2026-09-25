#!/usr/bin/env python3
"""Input-usage ablation (Tatarchenko et al. 2019, see notes/05).

Do our single-view models actually *use* the input image, or do they mostly
emit a category-centroid shape? We evaluate each model's usual test protocol
under three input conditions:

- ``real``     : the actual input image (the normal score)
- ``constant`` : a flat gray image (zeroed after normalization) — no object info
- ``shuffled`` : each object gets *another* object's image (image info, wrong object)

If ``constant``/``shuffled`` score close to ``real``, the model barely reads
the image. Results go to ``results/ablation/image_usage.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def _io(sample_pred, sample_gt, t):
    pb = sample_pred >= t
    gb = sample_gt >= 0.5
    u = int((pb | gb).sum())
    return float((pb & gb).sum()) / u if u else 1.0


# ------------------------------------------------------------------ Pix2Vox
def ablate_pix2vox(run_id: str, batch: int = 8):
    P2V = ROOT / "Pix2Vox"
    sys.path.insert(0, str(P2V))
    os_chdir = os.getcwd()
    os.chdir(P2V)
    from config import cfg
    import utils.data_loaders
    import utils.data_transforms
    import utils.network_utils
    from models.encoder import Encoder
    from models.decoder import Decoder
    from models.refiner import Refiner
    from models.merger import Merger

    ckpt_path = P2V / "output" / "checkpoints" / run_id / "best-ckpt.pth"
    cfg.CONST.WEIGHTS = str(ckpt_path)
    cfg.CONST.BATCH_SIZE = 1

    img_size = (cfg.CONST.IMG_H, cfg.CONST.IMG_W)
    crop = (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
    tf = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(img_size, crop),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    ds = loader.get_dataset(utils.data_loaders.DatasetType.TEST,
                            cfg.CONST.N_VIEWS_RENDERING, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1,
                                     pin_memory=True, shuffle=False)

    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
    ck = torch.load(str(ckpt_path), weights_only=False, map_location="cuda")
    epoch_idx = ck["epoch_idx"]
    encoder.load_state_dict(ck["encoder_state_dict"])
    decoder.load_state_dict(ck["decoder_state_dict"])
    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(ck["refiner_state_dict"])
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(ck["merger_state_dict"])
    for m in (encoder, decoder, refiner, merger):
        m.eval()

    imgs, gts = [], []
    for _, _, rendering_images, gt_volume in dl:
        imgs.append(rendering_images.clone())
        gts.append(gt_volume.clone())
    n = len(imgs)
    print(f"[pix2vox] {n} test samples loaded")

    def predict(ri):
        ri = utils.network_utils.var_or_cuda(ri)
        with torch.no_grad():
            feats = encoder(ri)
            raw, gen = decoder(feats)
            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                gen = merger(raw, gen)
            else:
                gen = torch.mean(gen, dim=1)
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                gen = refiner(gen)
        return gen

    def scores(image_list, t):
        vals = []
        for i in range(0, n, batch):
            b = torch.cat(image_list[i:i + batch], dim=0)
            pred = predict(b)
            for j in range(pred.shape[0]):
                p = pred[j]
                if p.dim() == 4:
                    p = p[0]
                g = gts[i + j]
                if g.dim() == 4:
                    g = g[0]
                vals.append(_io(p.cpu(), g.cpu(), t))
        return float(np.mean(vals))

    variants = {
        "real": imgs,
        "constant": [torch.zeros_like(im) for im in imgs],
        "shuffled": [imgs[(i + 1) % n] for i in range(n)],
    }
    out = {}
    for name, images in variants.items():
        out[name] = {"iou_t0.3": scores(images, 0.3), "iou_t0.5": scores(images, 0.5)}
        print(f"[pix2vox] {name:10s} IoU@0.3={out[name]['iou_t0.3']:.4f} "
              f"@0.5={out[name]['iou_t0.5']:.4f}")
    os.chdir(os_chdir)
    return out


# ----------------------------------------------------------------- AtlasNet
def ablate_atlasnet(run: str):
    ATL = ROOT / "AtlasNet"
    sys.path.insert(0, str(ATL))
    os.chdir(ATL)
    from easydict import EasyDict
    from model.model import EncoderDecoder
    from dataset.pointcloud_processor import Normalization
    from PIL import Image
    from torchvision import transforms as tv
    from scipy.spatial import cKDTree

    run_dir = ATL / "log" / run
    opt = EasyDict(json.loads((run_dir / "options.json").read_text()))
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = EncoderDecoder(opt)
    net = torch.nn.DataParallel(net, device_ids=[0]).to(opt.device)
    net.load_state_dict(torch.load(run_dir / "network.pth",
                                   map_location=opt.device, weights_only=False))
    net.eval()

    splits = json.loads((ATL / "dataset" / "data" / "splits.json").read_text())
    to_tensor = tv.Compose([tv.Resize(224, interpolation=2), tv.ToTensor()])
    center = tv.CenterCrop(127)

    samples = []
    for cat, sp in splits.items():
        for obj in sp.get("test", []):
            pc = ATL / "dataset" / "data" / "ShapeNetV1PointCloud" / cat / f"{obj}.npy"
            img = ATL / "dataset" / "data" / "ShapeNetV1Renderings" / cat / obj / "rendering" / "00.png"
            if pc.is_file() and img.is_file():
                samples.append((pc, img))
    print(f"[atlasnet] {len(samples)} test samples loaded")

    gts, imgs = [], []
    for pc, img in samples:
        pts = torch.from_numpy(np.load(pc)).float()
        gt = Normalization.normalize_unitL2ball_functional(pts[:, :3]).squeeze(0).numpy()
        gts.append(gt)
        imgs.append(to_tensor(center(Image.open(img)))[:3])
    n = len(imgs)

    def measure(image_list):
        chs, fss = [], []
        for i, im in enumerate(image_list):
            with torch.no_grad():
                out = net(im[None].to(opt.device), train=False)
            pred = out.transpose(2, 3).reshape(1, -1, 3)[0].cpu().numpy()
            gt = gts[i]
            da, _ = cKDTree(gt).query(pred)
            db, _ = cKDTree(pred).query(gt)
            chs.append(float(da.mean() + db.mean()))
            tau = 0.01
            precision = float((db < tau).mean())
            recall = float((da < tau).mean())
            fss.append(0.0 if precision + recall == 0 else
                       2 * precision * recall / (precision + recall))
        return float(np.mean(chs)), float(np.mean(fss))

    variants = {
        "real": imgs,
        "constant": [torch.zeros_like(im) for im in imgs],
        "shuffled": [imgs[(i + 1) % n] for i in range(n)],
    }
    out = {}
    for name, images in variants.items():
        ch, fs = measure(images)
        out[name] = {"chamfer": ch, "fscore": fs}
        print(f"[atlasnet] {name:10s} chamfer={ch:.4f} F={fs:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p2v-run-id", default="2026-09-25T21:38:11.694424")
    ap.add_argument("--atlas-run", default="svr25_pretrained")
    ap.add_argument("--model", choices=["both", "pix2vox", "atlasnet"], default="both")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablation" / "image_usage.json")
    args = ap.parse_args()

    result = {
        "question": "Does the model use the input image (Tatarchenko et al., CVPR 2019)?",
        "note": "constant = gray image (zero after normalization); "
                "shuffled = another object's image. Closing the gap to 'real' "
                "means the model ignores image content.",
    }
    if args.model in ("both", "pix2vox"):
        result["pix2vox"] = {"run": args.p2v_run_id,
                             **ablate_pix2vox(args.p2v_run_id)}
    if args.model in ("both", "atlasnet"):
        result["atlasnet"] = {"run": args.atlas_run,
                              **ablate_atlasnet(args.atlas_run)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
