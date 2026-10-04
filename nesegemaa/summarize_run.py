#!/usr/bin/env python3
"""Build results/atlasnet/<run>/summary.json from a finished training run.

Reads AtlasNet/log/<run>/{log.txt,options.json,train_stdout.log} and writes
the committed summary (same schema as the A0a/A0b summaries). Weight
provenance comes from the training stdout ([resnet] line, transplant line);
a missing provenance line is recorded as MISSING, never invented.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    src = ROOT / "AtlasNet" / "log" / args.run
    dst = ROOT / "results" / "atlasnet" / args.run
    log_lines = (src / "log.txt").read_text().splitlines()
    stdout = (src / "train_stdout.log").read_text().splitlines() \
        if (src / "train_stdout.log").is_file() else []
    st = [json.loads(l.split("json_stats:", 1)[1]) for l in log_lines
          if l.startswith("json_stats:")]
    if not st:
        raise RuntimeError(f"no json_stats epochs in {src / 'log.txt'}")
    lv = [s["loss_val"] for s in st]
    fs = [s["fscore"] for s in st]
    bi = min(range(len(lv)), key=lambda i: lv[i])
    fi = max(range(len(fs)), key=lambda i: fs[i])
    opt = json.loads((src / "options.json").read_text())

    resnet_lines = [l for l in stdout if "matched=" in l and "resnet" in l]
    transplant = [l for l in stdout if "750/750" in l or "reload-decoder" in l]

    rd_path = opt.get("reload_decoder_path") or ""
    if rd_path:
        dec = {"source": "shapenet-warmstart", "file": rd_path,
               "transplant": next(iter(transplant), "MISSING"),
               "sha256": sha256_file(rd_path) if Path(rd_path).is_file() else "FILE-GONE"}
    else:
        dec = {"source": "scratch"}
    summary = {
        "best_chamfer": lv[bi], "best_fscore": fs[fi],
        "final_chamfer": lv[-1], "final_fscore": fs[-1],
        "n_epochs": len(st), "best_chamfer_epoch": bi + 1,
        "best_fscore_epoch": fi + 1,
        "weights": {
            "encoder": {"source": "imagenet",
                        "provenance_log": next(iter(resnet_lines), "MISSING")},
            "decoder": dec,
        },
        "split": "splits.json test lists (70/15/15, seed 0)", "tau_squared": 0.001,
        "batch_size": opt.get("batch_size"),
        "batch_size_test": opt.get("batch_size_test"),
        "nb_primitives": opt.get("nb_primitives"),
        "template": opt.get("template_type"),
        "n_views": opt.get("n_views", 1), "views_pool": opt.get("views_pool", "none"),
        "number_points": opt.get("number_points"),
        "number_points_eval": opt.get("number_points_eval"),
        "bottleneck_size": opt.get("bottleneck_size"),
        "lrate": opt.get("lrate"),
        "lr_decay": [opt.get("lr_decay_1"), opt.get("lr_decay_2"), opt.get("lr_decay_3")],
        "freeze_encoder": bool(opt.get("freeze_encoder", False)),
        "loss_reg": opt.get("loss_reg", "none"),
        "loss_reg_w": opt.get("loss_reg_w", 0.0),
        "aug": sorted(k for k in ("random_rotation", "data_augmentation_axis_rotation",
                                  "data_augmentation_random_flips", "anisotropic_scaling",
                                  "random_translation") if opt.get(k)),
        "seed": opt.get("seed", -1),
    }
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("log.txt", "options.json", "curve.png", "curve_log.png"):
        if (src / n).is_file():
            import shutil
            shutil.copy2(src / n, dst / n)
    (dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary for {args.run}: best_ch={lv[bi]:.4f}@ep{bi+1} "
          f"best_f={fs[fi]:.4f}@ep{fi+1} ({len(st)} epochs)")
    print(json.dumps(summary["weights"], indent=2))


if __name__ == "__main__":
    main()
