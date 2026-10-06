#!/usr/bin/env python3
"""Visualize one benchmark object, point cloud, and optional rendered views.

Examples:
    python visualize_object.py --object battery/battery_001
    python visualize_object.py --object battery/battery_001 \
        --prediction results/3dgs/battery_battery_001.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".xyz", ".txt"}:
        points = np.loadtxt(path)
    elif path.suffix.lower() == ".obj":
        points = np.asarray(
            [
                [float(parts[1]), float(parts[2]), float(parts[3])]
                for line in path.read_text().splitlines()
                if (parts := line.split()) and parts[0] == "v" and len(parts) >= 4
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("Prediction must be .npy, .xyz, .txt, or .obj")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected prediction with shape (N, >=3), got {points.shape}")
    return points[:, :3]


def normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (points - center) / scale


def dataset_to_scene(points: np.ndarray) -> np.ndarray:
    """Convert OmniObject3D point-cloud axes (Y-up) to render axes (Z-up)."""
    return points[:, [0, 2, 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="category/object_id")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../benchmark/results/benchmark_v1.json"),
    )
    parser.add_argument("--prediction", type=Path)
    parser.add_argument(
        "--mesh-prediction",
        type=Path,
        help="Mesh OBJ to display alongside the point prediction.",
    )
    parser.add_argument(
        "--mesh-label",
        default="MESH VERTICES",
        help="Label for the mesh prediction in the comparison figure.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="Directory containing predicted PNGs named like 020.png, 021.png, ...",
    )
    parser.add_argument(
        "--prediction-space",
        choices=("scene", "raw"),
        default="scene",
        help="Coordinate space of point prediction: scene (default) or raw point-cloud units.",
    )
    parser.add_argument(
        "--mesh-space",
        choices=("scene", "dataset"),
        default="scene",
        help="Coordinate space of the OBJ vertices (current Pixel2Mesh outputs scene axes).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=20_000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    record = next(
        (
            item
            for item in manifest["objects"]
            if f'{item["category"]}/{item["object_id"]}' == args.object
        ),
        None,
    )
    if record is None:
        raise SystemExit(f"Object not found in manifest: {args.object}")

    dataset_dir = Path(manifest["dataset_dir"])
    cloud_path = dataset_dir / record["point_cloud"]
    target = np.load(cloud_path, allow_pickle=False).astype(np.float32)
    center = np.asarray(record["normalization"]["center"], dtype=np.float32)
    scale = float(record["normalization"]["scale"])
    # The trainer and camera poses use the dataset scene convention, whose
    # object bounds are approximately [-0.4, 0.4].
    target = dataset_to_scene(normalize(target, center, scale) * 0.4)

    rng = np.random.default_rng(0)
    if len(target) > args.max_points:
        target = target[rng.choice(len(target), args.max_points, replace=False)]

    prediction = None
    if args.prediction:
        prediction = load_prediction(args.prediction)
        if args.prediction_space == "raw":
            prediction = normalize(prediction, center, scale) * 0.4
        if len(prediction) > args.max_points:
            prediction = prediction[
                rng.choice(len(prediction), args.max_points, replace=False)
            ]
    mesh_prediction = None
    if args.mesh_prediction:
        mesh_prediction = load_prediction(args.mesh_prediction)
        if args.mesh_space == "dataset":
            mesh_prediction = dataset_to_scene(mesh_prediction)
        if len(mesh_prediction) > args.max_points:
            mesh_prediction = mesh_prediction[
                rng.choice(len(mesh_prediction), args.max_points, replace=False)
            ]

    render_dir = dataset_dir / record["render_dir"]
    image_paths = [
        render_dir / name
        for name in record["input_views"][:6]
        if (render_dir / name).is_file()
    ]
    heldout_names = record["heldout_views"]
    predicted_render_paths = (
        {
            name: args.render_dir / name
            for name in heldout_names
            if (args.render_dir / name).is_file()
        }
        if args.render_dir
        else {}
    )
    columns = max(6, 2 * len(heldout_names))
    rows = 3 if prediction is not None or mesh_prediction is not None else 2
    if predicted_render_paths:
        rows += 1
    figure = plt.figure(figsize=(3.5 * columns, 4 * rows))
    grid = figure.add_gridspec(rows, columns)
    figure.suptitle(f'{args.object} ({record["split"]})', fontsize=16)

    for index, image_path in enumerate(image_paths):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(Image.open(image_path))
        axis.set_title(f"INPUT / GT {image_path.stem}")
        axis.axis("off")

    target_axis = figure.add_subplot(grid[1, 0], projection="3d")
    target_axis.scatter(
        target[:, 0], target[:, 1], target[:, 2], s=1, alpha=0.7, c="tab:blue"
    )
    target_axis.set_title("GROUND TRUTH (blue)")

    if prediction is not None:
        prediction_axis = figure.add_subplot(grid[1, 1], projection="3d")
        prediction_axis.scatter(
            prediction[:, 0],
            prediction[:, 1],
            prediction[:, 2],
            s=1,
            alpha=0.7,
            c="tab:orange",
        )
        prediction_axis.set_title("GAUSSIAN CENTERS (orange)")
        overlay_axis = figure.add_subplot(grid[1, 2], projection="3d")
        overlay_axis.scatter(
            target[:, 0], target[:, 1], target[:, 2], s=1, alpha=0.25, c="tab:blue"
        )
        overlay_axis.scatter(
            prediction[:, 0],
            prediction[:, 1],
            prediction[:, 2],
            s=1,
            alpha=0.45,
            c="tab:orange",
        )
        overlay_axis.set_title("OVERLAY (blue GT / orange GS)")
    if mesh_prediction is not None:
        mesh_axis = figure.add_subplot(grid[1, 3], projection="3d")
        mesh_axis.scatter(
            mesh_prediction[:, 0],
            mesh_prediction[:, 1],
            mesh_prediction[:, 2],
            s=1,
            alpha=0.7,
            c="tab:green",
        )
        mesh_axis.set_title(f"{args.mesh_label} (green)")
        comparison_axis = figure.add_subplot(grid[1, 4], projection="3d")
        comparison_axis.scatter(
            target[:, 0], target[:, 1], target[:, 2], s=1, alpha=0.2, c="tab:blue"
        )
        if prediction is not None:
            comparison_axis.scatter(
                prediction[:, 0],
                prediction[:, 1],
                prediction[:, 2],
                s=1,
                alpha=0.35,
                c="tab:orange",
            )
        comparison_axis.scatter(
            mesh_prediction[:, 0],
            mesh_prediction[:, 1],
            mesh_prediction[:, 2],
            s=1,
            alpha=0.4,
            c="tab:green",
        )
        comparison_axis.set_title(
            "OVERLAY (GT / GS / MESH)" if prediction is not None else "OVERLAY (GT / MESH)"
        )

    if predicted_render_paths:
        render_row = rows - 1
        for index, name in enumerate(heldout_names):
            image_path = render_dir / name
            prediction_path = predicted_render_paths.get(name)
            if not image_path.is_file() or prediction_path is None:
                continue
            gt_axis = figure.add_subplot(grid[render_row, index * 2])
            gt_axis.imshow(Image.open(image_path))
            gt_axis.set_title(f"HELD-OUT GT {Path(name).stem}")
            gt_axis.axis("off")
            pred_axis = figure.add_subplot(grid[render_row, index * 2 + 1])
            pred_axis.imshow(Image.open(prediction_path))
            pred_axis.set_title(f"3DGS RENDER {Path(name).stem}")
            pred_axis.axis("off")

    for axis in figure.axes:
        if hasattr(axis, "zaxis"):
            axis.set_box_aspect((1, 1, 1))

    output = args.output or Path(
        f"results/qualitative/{args.object.replace('/', '_')}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
