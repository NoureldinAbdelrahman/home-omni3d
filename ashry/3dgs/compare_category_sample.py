#!/usr/bin/env python3
"""Convert and visualize every object in a 3DGS category-sample run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/category_sample"))
    parser.add_argument("--output-dir", type=Path, default=Path("../tools/qualitative/category_sample_3dgs"))
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--level", type=float, default=0.08)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    visualizer = root.parent / "tools" / "visualize_object.py"
    converter = root / "convert_mesh.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    record_by_directory = {
        f'{record["category"]}_{record["object_id"]}': record
        for record in manifest["objects"]
    }
    directories = sorted(path for path in args.results_dir.iterdir() if (path / "checkpoint.pt").is_file())
    for directory in directories:
        name = directory.name
        record = record_by_directory.get(name)
        if record is None:
            raise ValueError(f"Could not map result directory to manifest object: {name}")
        object_name = f'{record["category"]}/{record["object_id"]}'
        mesh = directory / "3dgs_mesh.obj"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--checkpoint",
                str(directory / "checkpoint.pt"),
                "--output",
                str(mesh),
                "--resolution",
                str(args.resolution),
                "--level",
                str(args.level),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(visualizer),
                "--manifest",
                str(args.manifest),
                "--object",
                object_name,
                "--prediction",
                str(directory / "gaussian_centers.npy"),
                "--mesh-prediction",
                str(mesh),
                "--mesh-label",
                "3DGS MESH VERTICES",
                "--render-dir",
                str(directory / "heldout"),
                "--output",
                str(args.output_dir / f"{name}.png"),
            ],
            check=True,
        )
    print(f"Processed {len(directories)} objects")


if __name__ == "__main__":
    main()
