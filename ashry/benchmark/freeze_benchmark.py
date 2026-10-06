#!/usr/bin/env python3
"""Create a reproducible object-level benchmark manifest.

The manifest is shared by all reconstruction models. It freezes object-level
splits, input/held-out camera views, and point-cloud normalization. It does
not copy the dataset, so the dataset may remain a symlink.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_view_range(value: str) -> list[int]:
    """Parse ``000-019`` or a comma-separated list such as ``000,004,008``."""
    value = value.strip()
    if "-" in value and "," not in value:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        if start > end:
            raise ValueError(f"Invalid descending view range: {value!r}")
        return list(range(start, end + 1))
    views = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not views:
        raise ValueError(f"No views in {value!r}")
    if len(set(views)) != len(views):
        raise ValueError(f"Duplicate view in {value!r}")
    return views


def split_ids(ids: list[str], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[str]]:
    """Split one category deterministically while keeping its objects together."""
    if not ids:
        return {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        counts = (1, 0, 0)
    elif n == 2:
        counts = (1, 0, 1)
    else:
        n_val = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        n_train = n - n_val - n_test
        while n_train < 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
            n_train = n - n_val - n_test
        counts = (n_train, n_val, n_test)

    n_train, n_val, n_test = counts
    train_end = n_train
    val_end = train_end + n_val
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:val_end + n_test]),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_object(
    dataset_dir: Path,
    category: str,
    object_id: str,
    required_views: set[int],
) -> dict[str, Any] | None:
    pc_path = dataset_dir / "point_clouds" / category / f"{object_id}.npy"
    render_dir = dataset_dir / "renders" / category / object_id
    if not pc_path.is_file() or not render_dir.is_dir():
        return None
    views = {
        int(path.stem)
        for path in render_dir.glob("*.png")
        if path.stem.isdigit()
    }
    if not required_views.issubset(views):
        return None

    try:
        points = np.load(pc_path, allow_pickle=False).astype(np.float32)
    except (OSError, ValueError):
        return None
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
        or not np.isfinite(points).all()
    ):
        return None

    # Canonical form: centroid at zero and furthest point at radius one.
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-8:
        return None

    return {
        "category": category,
        "object_id": object_id,
        "point_cloud": str(pc_path.relative_to(dataset_dir)),
        "render_dir": str(render_dir.relative_to(dataset_dir)),
        "point_count": int(points.shape[0]),
        "point_cloud_sha256": file_sha256(pc_path),
        "normalization": {
            "type": "centroid_unit_sphere",
            "center": [float(value) for value in center],
            "scale": scale,
        },
    }


def build_manifest(
    dataset_dir: Path,
    seed: int,
    input_views: list[int],
    heldout_views: list[int],
    ratios: tuple[float, float, float],
) -> dict[str, Any]:
    if set(input_views) & set(heldout_views):
        raise ValueError("Input and held-out views overlap")
    required_views = set(input_views) | set(heldout_views)
    point_root = dataset_dir / "point_clouds"
    render_root = dataset_dir / "renders"
    if not point_root.is_dir() or not render_root.is_dir():
        raise FileNotFoundError(
            f"Expected point_clouds/ and renders/ under {dataset_dir}"
        )

    objects_by_category: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for category_dir in sorted(point_root.iterdir()):
        if not category_dir.is_dir():
            continue
        valid: list[dict[str, Any]] = []
        for pc_path in sorted(category_dir.glob("*.npy")):
            item = inspect_object(
                dataset_dir, category_dir.name, pc_path.stem, required_views
            )
            if item is None:
                skipped += 1
            else:
                valid.append(item)
        if valid:
            objects_by_category[category_dir.name] = valid

    objects: list[dict[str, Any]] = []
    splits = {"train": [], "val": [], "test": []}
    for category_index, category in enumerate(sorted(objects_by_category)):
        items = objects_by_category[category]
        ids = [item["object_id"] for item in items]
        category_splits = split_ids(ids, seed + category_index, ratios)
        by_id = {item["object_id"]: item for item in items}
        for split_name, split_ids_for_category in category_splits.items():
            for object_id in split_ids_for_category:
                item = dict(by_id[object_id])
                item["split"] = split_name
                item["input_views"] = [f"{view:03d}.png" for view in input_views]
                item["heldout_views"] = [f"{view:03d}.png" for view in heldout_views]
                objects.append(item)
                splits[split_name].append(f"{category}/{object_id}")

    objects.sort(key=lambda item: (item["category"], item["object_id"]))
    for split in splits.values():
        split.sort()
    return {
        "format_version": 1,
        "seed": seed,
        "dataset_dir": str(dataset_dir),
        "split_ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "views": {
            "input": [f"{view:03d}.png" for view in input_views],
            "heldout": [f"{view:03d}.png" for view in heldout_views],
        },
        "target": {
            "source": "point_cloud",
            "normalization": "centroid_unit_sphere",
            "evaluation_points": 100_000,
        },
        "counts": {
            "objects": len(objects),
            "categories": len(objects_by_category),
            "skipped_invalid_objects": skipped,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "splits": splits,
        "objects": objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("../dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison/benchmark_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-views", default="000-019")
    parser.add_argument("--heldout-views", default="020-023")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        parser.error("train/val/test ratios must be positive and sum to 1")
    manifest = build_manifest(
        args.dataset_dir.resolve(),
        args.seed,
        parse_view_range(args.input_views),
        parse_view_range(args.heldout_views),
        ratios,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
