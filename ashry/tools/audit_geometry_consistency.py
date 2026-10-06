#!/usr/bin/env python3
"""Audit whether OmniObject3D point clouds match their calibrated renders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
from skimage.draw import polygon


def scene_cloud(manifest: dict, record: dict) -> np.ndarray:
    cloud = np.load(
        Path(manifest["dataset_dir"]) / record["point_cloud"], allow_pickle=False
    ).astype(np.float32)
    normalization = record["normalization"]
    cloud = (
        cloud - np.asarray(normalization["center"], dtype=np.float32)
    ) / float(normalization["scale"]) * 0.4
    return cloud[:, [0, 2, 1]]


def cameras(render_dir: Path, names: list[str], size: int) -> tuple[list[np.ndarray], np.ndarray]:
    metadata = json.loads((render_dir / "transforms.json").read_text())
    focal = 0.5 * size / np.tan(float(metadata["camera_angle_x"]) * 0.5)
    intrinsic = np.array(
        [[focal, 0, size * 0.5], [0, focal, size * 0.5], [0, 0, 1]],
        dtype=np.float32,
    )
    viewmats = []
    frames = {Path(frame["file_path"]).name: frame for frame in metadata["frames"]}
    for name in names:
        matrix = np.asarray(frames[name]["transform_matrix"], dtype=np.float32)
        matrix[:3, 1:3] *= -1
        viewmats.append(np.linalg.inv(matrix))
    return viewmats, intrinsic


def project(points: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray, size: int) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    camera = (viewmat @ homogeneous.T).T[:, :3]
    pixels = (intrinsic @ camera.T).T
    pixels = pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-6)
    valid = (
        (camera[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < size)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < size)
    )
    return pixels[valid]


def hull_mask(points: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if len(points) < 3:
        return mask
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    rows, cols = polygon(vertices[:, 1], vertices[:, 0], shape=mask.shape)
    mask[rows, cols] = True
    return mask


def render_mask(path: Path, size: int) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB").resize((size, size)))
    return image.max(axis=2) > 10


def metrics(projected: np.ndarray, target: np.ndarray, size: int, dilation: int) -> dict[str, float]:
    point_mask = np.zeros((size, size), dtype=bool)
    if len(projected):
        xy = np.rint(projected).astype(int)
        point_mask[xy[:, 1], xy[:, 0]] = True
    point_mask = binary_dilation(point_mask, iterations=dilation)
    hull = hull_mask(projected, size)
    intersection = np.logical_and(hull, target).sum()
    union = np.logical_or(hull, target).sum()
    point_hits = target[np.rint(projected).astype(int)[:, 1], np.rint(projected).astype(int)[:, 0]].mean() if len(projected) else 0.0
    return {
        "projected_points": int(len(projected)),
        "point_foreground_fraction": float(point_hits),
        "dilated_point_iou": float(np.logical_and(point_mask, target).sum() / max(np.logical_or(point_mask, target).sum(), 1)),
        "hull_iou": float(intersection / max(union, 1)),
        "render_foreground_fraction": float(target.mean()),
        "projected_hull_fraction": float(hull.mean()),
    }


def selected_records(manifest: dict, one_per_category: bool, limit: int | None) -> list[dict]:
    records = [record for record in manifest["objects"] if record["split"] == "val"]
    if one_per_category:
        seen, selected = set(), []
        for record in records:
            if record["category"] not in seen:
                seen.add(record["category"])
                selected.append(record)
        records = selected
    return records[:limit] if limit is not None else records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/geometry_audit"))
    parser.add_argument("--one-per-category", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--views", choices=("input", "heldout", "all"), default="all")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    records = selected_records(manifest, args.one_per_category, args.limit)
    dataset_dir = Path(manifest["dataset_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        render_dir = dataset_dir / record["render_dir"]
        if args.views == "input":
            names = record["input_views"]
        elif args.views == "heldout":
            names = record["heldout_views"]
        else:
            names = record["input_views"] + record["heldout_views"]
        viewmats, intrinsic = cameras(render_dir, names, args.size)
        points = scene_cloud(manifest, record)
        object_name = f'{record["category"]}_{record["object_id"]}'
        columns = min(6, len(names))
        rows_count = (len(names) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows_count, columns, figsize=(3 * columns, 3 * rows_count), squeeze=False
        )
        for index, (name, viewmat) in enumerate(zip(names, viewmats)):
            projected = project(points, viewmat, intrinsic, args.size)
            target = render_mask(render_dir / name, args.size)
            result = metrics(projected, target, args.size, args.dilation)
            row = {"object": f'{record["category"]}/{record["object_id"]}', "view": name, **result}
            rows.append(row)
            axis = axes[index // columns, index % columns]
            image = np.asarray(Image.open(render_dir / name).convert("RGB").resize((args.size, args.size)))
            axis.imshow(image)
            axis.scatter(projected[:, 0], projected[:, 1], s=1, c="lime", alpha=0.4)
            axis.set_title(f'{name} IoU={result["hull_iou"]:.2f}')
            axis.axis("off")
        for axis in axes.flat[len(names):]:
            axis.axis("off")
        figure.suptitle(f"Point-cloud/render consistency: {record['category']}/{record['object_id']}")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{object_name}.png", dpi=140)
        plt.close(figure)

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for object_name in sorted({row["object"] for row in rows}):
        object_rows = [row for row in rows if row["object"] == object_name]
        summary.append(
            {
                "object": object_name,
                "mean_point_foreground_fraction": float(np.mean([r["point_foreground_fraction"] for r in object_rows])),
                "mean_dilated_point_iou": float(np.mean([r["dilated_point_iou"] for r in object_rows])),
                "mean_hull_iou": float(np.mean([r["hull_iou"] for r in object_rows])),
            }
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Audited {len(records)} objects and {len(rows)} views")
    print(f"Wrote {csv_path} and {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
