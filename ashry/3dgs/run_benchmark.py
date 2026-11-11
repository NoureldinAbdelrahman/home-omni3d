#!/usr/bin/env python3
"""Train and evaluate 3DGS for every selected benchmark object.

The runner is resumable: completed objects with a checkpoint and metrics file
are skipped, while failed objects are recorded in the report and do not stop
the remaining batch unless --stop-on-error is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmark" / "results" / "benchmark_v1.json"
DEFAULT_ARTIFACTS = ROOT / "3dgs" / "results" / "benchmark_512_10k"
DEFAULT_REPORTS = ROOT / "3dgs_metrics" / "benchmark_512_10k"
TRAINER = Path(__file__).with_name("train_3dgs.py")


def select_records(manifest: dict, split: str, start: int, limit: int) -> list[dict]:
    records = manifest["objects"]
    if split != "all":
        records = [record for record in records if record["split"] == split]
    records = records[start:]
    return records if limit < 0 else records[:limit]


def aggregate(rows: list[dict]) -> dict[str, float | int]:
    numeric = [
        key for key, value in rows[0].items()
        if key not in {"object", "category", "split", "status", "error"}
        and isinstance(value, (int, float))
    ]
    result: dict[str, float | int] = {"completed": len(rows)}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_median"] = float(np.median(values))
    return result


def write_reports(output: Path, rows: list[dict], failures: list[dict]) -> None:
    total = len(rows) + len(failures)
    aggregate_metrics = aggregate(rows) if rows else {"completed": 0}
    aggregate_metrics.update({
        "failed": len(failures),
        "total_processed": total,
        "success_rate": float(len(rows) / total) if total else 0.0,
    })
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objects": rows,
        "failures": failures,
        "aggregate": aggregate_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = sorted({key for row in rows + failures for key in row})
    with (output / "benchmark_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="Temporary directory for checkpoints and renders; normally do not commit it.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORTS,
        help="Small, shareable directory for aggregate metrics.",
    )
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--visual-hull-candidates", type=int, default=150000)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse checkpoints that already have a metrics file.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--keep-pointclouds",
        action="store_true",
        help="Keep per-object PLY files after metrics are computed. They are not needed for reports.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    records = select_records(manifest, args.split, args.start, args.limit)
    rows: list[dict] = []
    failures: list[dict] = []

    # Import after argument parsing so --help works without importing torch.
    sys.path.insert(0, str(ROOT / "app"))
    from pointclouds import compare, export_3dgs

    for number, record in enumerate(records, start=args.start + 1):
        object_name = f'{record["category"]}/{record["object_id"]}'
        slug = object_name.replace("/", "_")
        object_dir = output_dir / slug
        checkpoint = object_dir / "checkpoint.pt"
        predicted_path = object_dir / "pointcloud.ply"
        metrics_path = object_dir / "pointcloud_metrics.json"
        print(f"[{number}/{args.start + len(records)}] {object_name}", flush=True)

        if args.skip_existing and checkpoint.is_file() and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
        else:
            object_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(TRAINER),
                "--manifest", str(manifest_path),
                "--object", object_name,
                "--image-size", str(args.image_size),
                "--iterations", str(args.iterations),
                "--save-every", str(args.save_every),
                "--visual-hull-candidates", str(args.visual_hull_candidates),
                "--output-dir", str(output_dir),
            ]
            try:
                subprocess.run(command, check=True)
                export_3dgs(checkpoint, predicted_path, args.point_count)
                metrics = compare(
                    "3dgs", predicted_path, manifest_path, object_name, metrics_path
                )
                if not args.keep_pointclouds:
                    predicted_path.unlink(missing_ok=True)
            except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
                failure = {
                    "object": object_name,
                    "category": record["category"],
                    "split": record["split"],
                    "status": "failed",
                    "error": str(error),
                }
                failures.append(failure)
                write_reports(report_dir, rows, failures)
                print(f"FAILED {object_name}: {error}", flush=True)
                if args.stop_on_error:
                    raise
                continue

        row = {
            "object": object_name,
            "category": record["category"],
            "split": record["split"],
            "status": "completed",
            **metrics,
        }
        rows.append(row)
        write_reports(report_dir, rows, failures)

    summary = aggregate(rows) if rows else {"completed": 0}
    summary.update({
        "failed": len(failures),
        "total_processed": len(rows) + len(failures),
        "success_rate": float(len(rows) / (len(rows) + len(failures)))
        if rows or failures else 0.0,
    })
    print(json.dumps(summary, indent=2))
    print(f"Reports: {report_dir / 'benchmark_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
