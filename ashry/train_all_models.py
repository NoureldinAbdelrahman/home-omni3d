#!/usr/bin/env python3
"""Run resumable 3DGS, NeRF, and NeuS training for the benchmark objects.

The default is deliberately sequential: three CUDA processes competing for one
GPU are normally slower and can exhaust VRAM. Existing checkpoints are skipped
so the command can safely be left running overnight and restarted later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "benchmark" / "results" / "benchmark_v1.json"


def run(command: list[str], label: str, log: Path, continue_on_error: bool) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started}] START {label}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{started}] START {label}\n")
        stream.flush()
        try:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            finished = time.strftime("%Y-%m-%d %H:%M:%S")
            stream.write(f"[{finished}] FAILED exit={error.returncode}\n")
            print(f"[{finished}] FAILED {label}; see {log}", flush=True)
            if not continue_on_error:
                raise
            return False
    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{finished}] DONE  {label}", flush=True)
    return True


def selected_records(manifest: dict, split: str, start: int, limit: int, one_per_category: bool) -> list[dict]:
    records = manifest["objects"]
    if split != "all":
        records = [record for record in records if record["split"] == split]
    if one_per_category:
        selected: list[dict] = []
        seen: set[str] = set()
        for record in records:
            if record["category"] not in seen:
                seen.add(record["category"])
                selected.append(record)
        records = selected
    records = records[start:]
    return records if limit < 0 else records[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--one-per-category", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--gs-iterations", type=int, default=6000)
    parser.add_argument("--nerf-iterations", type=int, default=6000)
    parser.add_argument("--neus-iterations", type=int, default=6000)
    parser.add_argument("--gs-output", type=Path, default=ROOT / "3dgs" / "results" / "batch_512_fast")
    parser.add_argument("--nerf-output", type=Path, default=ROOT / "nerf" / "results" / "batch_512_fast")
    parser.add_argument("--neus-output", type=Path, default=ROOT / "neus" / "results" / "batch_512_fast")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "overnight_training.log")
    parser.add_argument("--no-3dgs", action="store_true")
    parser.add_argument("--no-nerf", action="store_true")
    parser.add_argument("--no-neus", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with later objects after one training process fails.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.resolve().read_text())
    records = selected_records(manifest, args.split, args.start, args.limit, args.one_per_category)
    python = sys.executable
    jobs: list[tuple[str, Path, list[str]]] = []

    for record in records:
        object_name = f'{record["category"]}/{record["object_id"]}'
        slug = object_name.replace("/", "_")
        if not args.no_3dgs:
            output = args.gs_output / slug
            jobs.append(("3DGS", output, [
                python, str(ROOT / "3dgs" / "train_3dgs.py"),
                "--manifest", str(args.manifest.resolve()), "--object", object_name,
                "--image-size", str(args.image_size), "--iterations", str(args.gs_iterations),
                "--save-every", "1000", "--output-dir", str(args.gs_output.resolve()),
            ]))
        if not args.no_nerf:
            output = args.nerf_output / slug
            jobs.append(("NeRF", output, [
                python, str(ROOT / "nerf" / "train.py"),
                "--manifest", str(args.manifest.resolve()), "--object", object_name,
                "--iterations", str(args.nerf_iterations), "--image-size", str(args.image_size),
                "--rays-per-step", "1024", "--coarse-samples", "24", "--fine-samples", "24",
                "--render-chunk", "2048", "--output-dir", str(output),
            ]))
        if not args.no_neus:
            output = args.neus_output / slug
            jobs.append(("NeuS", output, [
                python, str(ROOT / "neus" / "train.py"),
                "--manifest", str(args.manifest.resolve()), "--object", object_name,
                "--iterations", str(args.neus_iterations), "--image-size", str(args.image_size),
                "--rays-per-step", "1024", "--coarse-samples", "32", "--fine-samples", "32",
                "--render-chunk", "2048", "--eikonal-points", "512",
                "--checkpoint-every", "1000", "--output-dir", str(output),
            ]))

    completed = failed = skipped = 0
    for method, output, command in jobs:
        checkpoint = output / "checkpoint.pt"
        if checkpoint.is_file():
            print(f"SKIP {method} {output.name}: checkpoint exists", flush=True)
            skipped += 1
            continue
        if run(command, f"{method} {output.name}", args.log.resolve(), args.continue_on_error):
            completed += 1
        else:
            failed += 1
    print(f"Finished: completed={completed} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
