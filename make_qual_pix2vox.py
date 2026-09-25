#!/usr/bin/env python3
"""Save Pix2Vox predictions vs ground-truth voxels on held-out test objects.

For a few test objects per category it writes, under
``results/qualitative/pix2vox/<cat>_<obj>/``: ``input.png`` (the single view),
``pred.npy`` / ``gt.npy`` (32^3 volumes) and ``metrics.json`` (IoU).  The
notebook renders these side by side against the expected shape.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.utils.data
from PIL import Image

ROOT = Path(__file__).resolve().parent
P2V = ROOT / "Pix2Vox"
sys.path.insert(0, str(P2V))
os.chdir(P2V)

from config import cfg  # noqa: E402
import utils.data_loaders  # noqa: E402
import utils.data_transforms  # noqa: E402
import utils.network_utils  # noqa: E402
from models.encoder import Encoder  # noqa: E402
from models.decoder import Decoder  # noqa: E402
from models.refiner import Refiner  # noqa: E402
from models.merger import Merger  # noqa: E402


def build_networks(ckpt):
    encoder, decoder, refiner, merger = Encoder(cfg), Decoder(cfg), Refiner(cfg), Merger(cfg)
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
    checkpoint = torch.load(str(ckpt), weights_only=False, map_location="cuda")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(checkpoint["refiner_state_dict"])
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(checkpoint["merger_state_dict"])
    encoder.eval(); decoder.eval(); refiner.eval(); merger.eval()
    return checkpoint["epoch_idx"], encoder, decoder, refiner, merger


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "qualitative" / "pix2vox")
    ap.add_argument("--categories", nargs="+",
                    default=["medicine_bottle", "cup", "teapot", "hammer", "shampoo"])
    args = ap.parse_args()

    ckpt = P2V / "output" / "checkpoints" / args.run_id / "best-ckpt.pth"
    cfg.CONST.WEIGHTS = str(ckpt)
    cfg.CONST.BATCH_SIZE = 1

    tax = json.loads(open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH).read())
    want = {t["taxonomy_id"]: t["test"][0] for t in tax
            if t["taxonomy_id"] in args.categories and t.get("test")}

    img_size = (cfg.CONST.IMG_H, cfg.CONST.IMG_W)
    crop = (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
    transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(img_size, crop),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    ds = loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, transforms)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    epoch_idx, encoder, decoder, refiner, merger = build_networks(ckpt)
    args.out.mkdir(parents=True, exist_ok=True)
    found = {}
    for taxonomy_id, sample_name, rendering_images, gt_volume in dl:
        tid = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sname = sample_name[0]
        if tid not in want or want[tid] != sname or tid in found:
            continue
        ri = utils.network_utils.var_or_cuda(rendering_images)
        gv = utils.network_utils.var_or_cuda(gt_volume)
        with torch.no_grad():
            feats = encoder(ri)
            raw, gen = decoder(feats)
            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                gen = merger(raw, gen)
            else:
                gen = torch.mean(gen, dim=1)
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                gen = refiner(gen)
        pred = gen[0].cpu().numpy()
        gt = gv[0].cpu().numpy()
        if pred.ndim == 4:
            pred, gt = pred[0], gt[0]

        view = ri[0, 0] if ri.dim() == 5 else ri[0]  # [C,H,W]
        img = (view.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        outd = args.out / f"{tid}_{sname}"
        outd.mkdir(parents=True, exist_ok=True)
        np.save(outd / "pred.npy", pred.astype(np.float32))
        np.save(outd / "gt.npy", gt.astype(np.float32))
        # Save a *displayable* input: the original render cropped tightly to
        # the object (the network input is a center crop that usually cuts
        # most of the object out, which looks confusing in the report).
        try:
            sys.path.insert(0, str(ROOT))
            from plotting import load_display_image
            load_display_image(
                ROOT / "dataset" / "renders" / tid / sname / "000.png"
            ).save(outd / "input.png")
        except Exception:
            Image.fromarray((img * 255).astype(np.uint8)).save(outd / "input.png")

        pb, gb = pred >= 0.3, gt >= 0.5
        inter = float((pb & gb).sum())
        union = float((pb | gb).sum())
        iou = inter / union if union else 0.0
        (outd / "metrics.json").write_text(json.dumps({
            "category": tid, "object": sname, "iou_t0.3": iou,
            "n_pred_voxels": int(pb.sum()), "n_gt_voxels": int(gb.sum()),
        }, indent=2) + "\n")
        found[tid] = iou
        print(f"  {tid:16s} {sname:20s} IoU@0.3={iou:.4f}")
        if len(found) == len(want):
            break
    print("wrote", args.out, "| IoUs:", found)


if __name__ == "__main__":
    main()
