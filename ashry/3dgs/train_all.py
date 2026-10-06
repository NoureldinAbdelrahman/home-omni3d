#!/usr/bin/env python3
"""Train image-only 3DGS reconstructions for a benchmark split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json")
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--limit", type=int, help="Train only the first N selected objects.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--one-per-category",
        action="store_true",
        help="Select at most one object from each category.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/batch"))
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip objects with an existing checkpoint.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create a qualitative GT/geometry/held-out comparison after each object.",
    )
    parser.add_argument("--visual-hull-candidates", type=int, default=150000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    records = manifest["objects"]
    if args.split != "all":
        records = [record for record in records if record["split"] == args.split]
    if args.one_per_category:
        selected = []
        categories = set()
        for record in records:
            if record["category"] not in categories:
                categories.add(record["category"])
                selected.append(record)
        records = selected
    records = records[args.start :]
    if args.limit is not None:
        records = records[: args.limit]

    trainer = Path(__file__).with_name("train_3dgs.py")
    for number, record in enumerate(records, start=args.start + 1):
        object_name = f'{record["category"]}/{record["object_id"]}'
        object_dir = args.output_dir / object_name.replace("/", "_")
        checkpoint = object_dir / "checkpoint.pt"
        if args.skip_existing and checkpoint.is_file():
            print(f"[{number}] skip {object_name}: checkpoint exists", flush=True)
            continue

        command = [
            sys.executable,
            str(trainer),
            "--manifest",
            str(args.manifest),
            "--object",
            object_name,
            "--image-size",
            str(args.image_size),
            "--iterations",
            str(args.iterations),
            "--save-every",
            str(args.save_every),
            "--visual-hull-candidates",
            str(args.visual_hull_candidates),
            "--output-dir",
            str(args.output_dir),
        ]
        print(f"[{number}] train {object_name}", flush=True)
        subprocess.run(command, check=True)
        if args.visualize:
            visualization = Path(__file__).parents[1] / "tools" / "visualize_object.py"
            qualitative = Path("qualitative") / object_name.replace("/", "_")
            subprocess.run(
                [
                    sys.executable,
                    str(visualization),
                    "--manifest",
                    str(args.manifest),
                    "--object",
                    object_name,
                    "--prediction",
                    str(object_dir / "gaussian_centers.npy"),
                    "--render-dir",
                    str(object_dir / "heldout"),
                    "--output",
                    str(qualitative.with_suffix(".png")),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
