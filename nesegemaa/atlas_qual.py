#!/usr/bin/env python3
"""AtlasNet-SVR qualitative + full-split scoring driver (nesegemaa track).

Supersedes the removed root ``make_qual_atlasnet.py`` (recoverable from tag
``milestone-1``): same I/O contract, plus what the old script cannot do:

- ``--weights {best,final}`` (default: ``best``) — loads ``best-model.pth``
  (best validation Chamfer, see ``AtlasNet/training/trainer_abstract.py``).
  Missing file is a hard error, never a silent fallback to the other file.
- ``--eval_views`` — comma-separated render indices (default ``0``; e.g.
  ``0,6,12,18`` for the view-sensitivity study). Same GT cloud every time.
- Uses the ``splits.json`` TEST lists (never the old positional 80/20 slice),
  so numbers are comparable across runs by construction.
- Scores EVERY test object (forward-only, milliseconds each) into
  ``results/atlasnet/<run>/qual_metrics.json`` + ``qual_agg.json``
  (micro/macro, noise flags); input/pred/gt triplets are written only for the
  ``--panel`` objects used in report figures.

Rules honored: reconstruction input is the IMAGE ONLY (GT clouds are used for
scoring, never fed to the network — asserted below); τ=0.01; UnitBall
normalization identical to training.
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
from plotting import load_display_image  # noqa: E402

TAU = 0.01
DEFAULT_PANEL = [
    ("medicine_bottle", "medicine_bottle_068"),
    ("cup", "cup_032"),
    ("teapot", "teapot_031"),
    ("hammer", "hammer_019"),
    ("shampoo", "shampoo_031"),
]


def chamfer_and_fscore(a, b, tau=TAU):
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
    ap.add_argument("--run", required=True, help="AtlasNet/log/<run> directory name")
    ap.add_argument("--weights", choices=["best", "final"], default="best",
                    help="best-model.pth (best val Chamfer) or network.pth (final epoch)")
    ap.add_argument("--eval_views", default="0",
                    help="comma-separated render indices, e.g. 0,6,12,18")
    ap.add_argument("--panel", nargs="+", default=None,
                    help="'cat:obj' pairs for figure triplets; default: the 5 MS1 objects "
                         "(falls back to the category's test[0] with a printed note)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "qualitative" / "atlasnet")
    args = ap.parse_args()

    views = [int(v) for v in args.eval_views.split(",") if v.strip() != ""]
    assert views, "--eval_views must list at least one view index"

    run_dir = ATL / "log" / args.run
    wname = "best-model.pth" if args.weights == "best" else "network.pth"
    wpath = run_dir / wname
    if not wpath.is_file():
        raise RuntimeError(
            f"{wpath} missing — refusing to silently fall back to the other "
            f"weights file. (Did this run save a best-model snapshot? "
            f"Requires the trainer best-epoch patch.)")
    opt = EasyDict(json.loads((run_dir / "options.json").read_text()))
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = EncoderDecoder(opt)
    net = torch.nn.DataParallel(net, device_ids=[0]).to(opt.device)
    net.load_state_dict(torch.load(wpath, map_location=opt.device, weights_only=False))
    net.eval()
    print(f"loaded {wname} from {run_dir} (strict)")

    splits = json.loads((ATL / "dataset" / "data" / "splits.json").read_text())
    test_lists = {c: sp.get("test", []) for c, sp in splits.items()}

    to_tensor = transforms.Compose([transforms.Resize(224, interpolation=2), transforms.ToTensor()])
    center = transforms.CenterCrop(127)

    def load_gt(cat, obj):
        pts = torch.from_numpy(np.load(
            ATL / "dataset" / "data" / "ShapeNetV1PointCloud" / cat / f"{obj}.npy")).float()
        return Normalization_wrap(pts)

    def Normalization_wrap(pts):
        return pp.Normalization.normalize_unitL2ball_functional(pts[:, :3]).squeeze(0).numpy()

    def predict(im):
        with torch.no_grad():
            out = net(im[None].to(opt.device), train=False)  # [1, prim, 3, P]
        return out.transpose(2, 3).reshape(1, -1, 3)[0].cpu().numpy()

    # ---- panel objects (figure triplets) ----
    if args.panel:
        panel = []
        for item in args.panel:
            c, o = item.split(":")
            panel.append((c, o))
    else:
        panel = []
        for c, o in DEFAULT_PANEL:
            if o in test_lists.get(c, []):
                panel.append((c, o))
            else:
                fb = test_lists.get(c, [None])[0]
                print(f"  note: MS1 panel object {c}:{o} not in test split -> using {c}:{fb}")
                if fb is not None:
                    panel.append((c, fb))

    per_object_rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for v in views:
        for cat, objs in sorted(test_lists.items()):
            for obj in objs:
                img_path = (ATL / "dataset" / "data" / "ShapeNetV1Renderings"
                            / cat / obj / "rendering" / f"{v:02d}.png")
                if not img_path.is_file():
                    print(f"  skip {cat}/{obj} view {v}: render missing")
                    continue
                im = to_tensor(center(Image.open(img_path)))[:3]
                assert im.shape[0] == 3  # image only; GT never enters recon
                pred = predict(im)
                gt = load_gt(cat, obj)
                chamfer, fscore = chamfer_and_fscore(pred, gt, tau=TAU)
                per_object_rows.append({
                    "category": cat, "object": obj, "view": v,
                    "chamfer": chamfer, "fscore_tau0.01": fscore,
                    "n_points": int(pred.shape[0]),
                })
                if (cat, obj) in panel and v == views[0]:
                    outd = args.out / f"{cat}_{obj}"
                    outd.mkdir(parents=True, exist_ok=True)
                    np.save(outd / "pred.npy", pred.astype(np.float32))
                    np.save(outd / "gt.npy", gt.astype(np.float32))
                    try:
                        load_display_image(img_path).save(outd / "input.png")
                    except Exception:
                        Image.fromarray(
                            (im.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        ).save(outd / "input.png")
                    (outd / "metrics.json").write_text(json.dumps({
                        "category": cat, "object": obj, "view": v,
                        "chamfer": chamfer, "fscore_tau0.01": fscore,
                        "n_points": int(pred.shape[0]),
                    }, indent=2) + "\n")

    # ---- aggregates (micro/macro over the shared test split) ----
    by_view = {}
    for v in views:
        rows = [r for r in per_object_rows if r["view"] == v]
        cats = {}
        for r in rows:
            cats.setdefault(r["category"], []).append(r)
        micro_ch = float(np.mean([r["chamfer"] for r in rows])) if rows else None
        micro_fs = float(np.mean([r["fscore_tau0.01"] for r in rows])) if rows else None
        cat_means = [(c, float(np.mean([r["chamfer"] for r in rs])),
                      float(np.mean([r["fscore_tau0.01"] for r in rs])), len(rs))
                     for c, rs in sorted(cats.items())]
        macro_ch = float(np.mean([m[1] for m in cat_means])) if cat_means else None
        macro_fs = float(np.mean([m[2] for m in cat_means])) if cat_means else None
        noise = sorted(c for c, _, _, n in cat_means if n < 3)
        by_view[str(v)] = {
            "n_objects": len(rows), "n_categories": len(cat_means),
            "micro_chamfer": micro_ch, "micro_fscore": micro_fs,
            "macro_chamfer": macro_ch, "macro_fscore": macro_fs,
            "noise_categories_lt3": noise,
            "per_category": [
                {"category": c, "n": n, "chamfer": ch, "fscore": fs}
                for c, ch, fs, n in cat_means],
        }
        print(f"  view {v:2d}: n={len(rows):3d} micro_ch={micro_ch:.4f} "
              f"micro_f={micro_fs:.4f} macro_ch={macro_ch:.4f} macro_f={macro_fs:.4f} "
              f"noise={noise if noise else '-'}")

    res_dir = ROOT / "results" / "atlasnet" / args.run
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "qual_metrics.json").write_text(json.dumps(per_object_rows, indent=2) + "\n")
    agg = {
        "run": args.run, "weights": wname, "views": views, "tau": TAU,
        "split": "splits.json test lists",
        "n_test_objects_total": len(per_object_rows),
        "by_view": by_view,
    }
    (res_dir / "qual_agg.json").write_text(json.dumps(agg, indent=2) + "\n")
    print("wrote", res_dir / "qual_agg.json", "+", args.out)


if __name__ == "__main__":
    main()
