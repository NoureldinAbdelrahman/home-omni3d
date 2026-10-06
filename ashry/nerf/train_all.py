#!/usr/bin/env python3
"""Train the image-only NeRF on the selected category-balanced sample."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--one-per-category", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rays-per-step", type=int, default=2048)
    parser.add_argument("--coarse-samples", type=int, default=32)
    parser.add_argument("--fine-samples", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/category_sample"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest["objects"] if r["split"] == "val"]
    if args.one_per_category:
        seen, selected = set(), []
        for record in records:
            if record["category"] not in seen:
                seen.add(record["category"])
                selected.append(record)
        records = selected
    if args.limit is not None:
        records = records[:args.limit]
    script = Path(__file__).resolve().parent / "train.py"
    for record in records:
        name = f'{record["category"]}/{record["object_id"]}'
        output = args.output_root / name.replace("/", "_")
        if args.skip_existing and (output / "checkpoint.pt").exists():
            print(f"Skipping existing {name}", flush=True)
            continue
        subprocess.run([sys.executable, str(script), "--manifest", str(args.manifest), "--object", name, "--iterations", str(args.iterations), "--image-size", str(args.image_size), "--rays-per-step", str(args.rays_per_step), "--coarse-samples", str(args.coarse_samples), "--fine-samples", str(args.fine_samples), "--output-dir", str(output)], check=True)


if __name__ == "__main__":
    main()
