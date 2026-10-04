#!/usr/bin/env python3
"""Append one row per AtlasNet run to nesegemaa/ablation_atlasnet.csv.

Reads results/atlasnet/<run>/{summary.json,qual_agg.json}. Missing qual data
writes empty metric cells rather than failing (status column tells the story).
Header is created once; reruns replace the row for the same run.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "nesegemaa" / "ablation_atlasnet.csv"
COLS = ["run", "K_views", "pool", "patches", "template", "npts_train", "npts_eval",
        "bottleneck", "encoder_init", "encoder_train", "aug", "loss", "loss_w",
        "lr", "batch", "seed", "nepoch", "best_chamfer", "best_chamfer_ep",
        "best_fscore", "best_fscore_ep", "final_chamfer", "final_fscore",
        "qual_micro_ch", "qual_micro_f", "qual_macro_ch", "qual_macro_f",
        "n_test", "status", "notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--status", default="OK")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    d = ROOT / "results" / "atlasnet" / args.run
    s = json.loads((d / "summary.json").read_text())
    qpath = d / "qual_agg.json"
    q = json.loads(qpath.read_text()) if qpath.is_file() else {}
    v0 = (q.get("by_view") or {}).get("0", {})
    w = s.get("weights", {})
    row = {
        "run": args.run,
        "K_views": s.get("n_views", 1), "pool": s.get("views_pool", "none"),
        "patches": s.get("nb_primitives"), "template": s.get("template"),
        "npts_train": s.get("number_points"), "npts_eval": s.get("number_points_eval"),
        "bottleneck": s.get("bottleneck_size"),
        "encoder_init": ("imagenet" if "imagenet" in json.dumps(w.get("encoder", {}))
                         else json.dumps(w.get("encoder", {}))[:40]),
        "encoder_train": ("frozen" if s.get("freeze_encoder") else "finetuned"),
        "aug": ",".join(s.get("aug", [])) or "none",
        "loss": ("chamfer+edge" if s.get("loss_reg") == "edge" else "chamfer"),
        "loss_w": s.get("loss_reg_w", 0.0),
        "lr": s.get("lrate"), "batch": s.get("batch_size"),
        "seed": s.get("seed", -1), "nepoch": s.get("n_epochs"),
        "best_chamfer": s.get("best_chamfer"), "best_chamfer_ep": s.get("best_chamfer_epoch"),
        "best_fscore": s.get("best_fscore"), "best_fscore_ep": s.get("best_fscore_epoch"),
        "final_chamfer": s.get("final_chamfer"), "final_fscore": s.get("final_fscore"),
        "qual_micro_ch": v0.get("micro_chamfer"), "qual_micro_f": v0.get("micro_fscore"),
        "qual_macro_ch": v0.get("macro_chamfer"), "qual_macro_f": v0.get("macro_fscore"),
        "n_test": v0.get("n_objects"), "status": args.status, "notes": args.notes,
    }
    rows = []
    if CSV.is_file():
        with open(CSV) as f:
            rows = [r for r in csv.DictReader(f) if r.get("run") != args.run]
    rows.append(row)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"ablation row for {args.run}: status={args.status}")


if __name__ == "__main__":
    main()
