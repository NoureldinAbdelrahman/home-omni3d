#!/usr/bin/env python3
"""Export comparable point clouds from the three native ashry checkpoints."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes

ASHRY = Path(__file__).resolve().parents[1]


def load_class(directory: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"ashry_{name.lower()}_model", directory / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


NeRF = load_class(ASHRY / "nerf", "NeRF")
NeuS = load_class(ASHRY / "neus", "NeuS")


def export_3dgs_mesh(checkpoint: Path, resolution: int = 96) -> dict:
    """Extract a compact vertex-colored surface for browser inspection."""
    spec = importlib.util.spec_from_file_location("ashry_mesh_converter", ASHRY / "3dgs" / "convert_mesh.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load 3DGS mesh converter")
    converter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter_module)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    density, bounds_min = converter_module.density_grid(data, resolution, 0.02, 3.0)
    from scipy import ndimage

    level = min(0.08, max(0.01, float(np.percentile(density[density > 0], 70))))
    occupied = density >= level
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        raise RuntimeError("3DGS density did not form a surface at the adaptive threshold.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    component = labels == largest
    if component.sum() < max(32, resolution):
        raise RuntimeError("3DGS density surface is too fragmented for a connected mesh.")
    smooth_density = ndimage.gaussian_filter(density * component, sigma=0.8)
    vertices, faces, _, _ = marching_cubes(smooth_density, level=level * 0.65)
    means = data["means"].numpy().astype(np.float32)
    colors = data["colors"][:, 0].numpy() * np.float32(0.2820947917738781) + 0.5
    colors = np.clip(colors, 0, 1)
    scales = np.exp(data["scales"].numpy()).clip(1e-4, 1.0)
    opacity = torch.sigmoid(data["opacities"]).numpy()
    valid = opacity >= 0.005
    means, colors, scales = means[valid], colors[valid], scales[valid]
    extent = float(np.max(density.shape) - 1)
    raw_min = (data["means"].numpy() - 3.0 * np.exp(data["scales"].numpy()).max(axis=1, keepdims=True)).min(axis=0) - 0.02
    raw_max = (data["means"].numpy() + 3.0 * np.exp(data["scales"].numpy()).max(axis=1, keepdims=True)).max(axis=0) + 0.02
    cube_extent = float(np.max(raw_max - raw_min))
    vertices = bounds_min + vertices * (cube_extent / extent)
    from scipy.spatial import cKDTree

    nearest = cKDTree(means).query(vertices, k=1, workers=-1)[1]
    vertex_colors = (colors[nearest] * 255).round().astype(np.uint8)
    return {
        "vertices": vertices.astype(np.float32).tolist(),
        "faces": faces.astype(np.int32).tolist(),
        "colors": vertex_colors.tolist(),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.full((len(points), 3), 190, dtype=np.uint8) if colors is None else np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")


def sample_grid(bounds: float, resolution: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-bounds, bounds, resolution, device=device)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)


def export_3dgs(
    checkpoint: Path,
    output: Path,
    limit: int,
    colors_output: Path | None = None,
) -> np.ndarray:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    points = data["means"].numpy().astype(np.float32)
    opacity = torch.sigmoid(data["opacities"]).numpy()
    keep = opacity >= 0.005
    points = points[keep]
    sh = data.get("colors")
    learned_colors = None
    if sh is not None:
        learned_colors = (
            sh[:, 0].numpy() * np.float32(0.2820947917738781) + 0.5
        ).clip(0, 1)
        learned_colors = (learned_colors[keep] * 255).round().astype(np.uint8)
    if len(points) > limit:
        # Keep a deterministic, spatially distributed sample with the same
        # cardinality as the benchmark cloud.
        indices = np.linspace(0, len(points) - 1, limit).astype(int)
        points = points[indices]
        if learned_colors is not None:
            learned_colors = learned_colors[indices]
    # Gaussian means are already in the scene frame used by transforms.json.
    # Do not apply the dataset Y-up to scene-axis conversion a second time.
    write_ply(output, points, np.full((len(points), 3), (100, 210, 240), dtype=np.uint8))
    if colors_output is not None:
        write_ply(
            colors_output,
            points,
            learned_colors
            if learned_colors is not None
            else np.full((len(points), 3), (170, 170, 170), dtype=np.uint8),
        )
    return points


def extract_surface(model: torch.nn.Module, output: Path, bounds: float, resolution: int, threshold: float, limit: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    grid = sample_grid(bounds, resolution, device)
    values = []
    with torch.no_grad():
        for chunk in grid.split(65536):
            if isinstance(model, NeRF):
                directions = torch.zeros_like(chunk)
                density, _ = model(chunk, directions)
                values.append(density[:, 0].cpu())
            else:
                sdf, _ = model.sdf_network(chunk)
                values.append(sdf[:, 0].cpu())
    field = torch.cat(values).reshape(resolution, resolution, resolution).numpy()
    if isinstance(model, NeRF):
        if field.max() <= threshold:
            raise RuntimeError(f"NeRF density never reaches threshold {threshold}")
        vertices, _, _, _ = marching_cubes(field, level=threshold)
        color = (255, 190, 105)
    else:
        if field.min() > 0 or field.max() < 0:
            raise RuntimeError("NeuS SDF grid does not cross zero")
        vertices, _, _, _ = marching_cubes(field, level=0.0)
        color = (200, 160, 255)
    points = -bounds + vertices * (2 * bounds / (resolution - 1))
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit).astype(int)]
    # The sampled field is evaluated in the trainer's scene frame.
    write_ply(output, points, np.full((len(points), 3), color, dtype=np.uint8))
    return points.astype(np.float32)


def export(method: str, checkpoint: Path, output: Path, resolution: int = 96, limit: int = 50000) -> np.ndarray:
    output.parent.mkdir(parents=True, exist_ok=True)
    if method == "3dgs":
        return export_3dgs(checkpoint, output, limit)
    if method == "nerf":
        return extract_surface(NeRF(), output, 1.0, resolution, 8.0, limit)
    if method == "neus":
        model = NeuS()
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(data["model"])
        return extract_surface(model, output, 1.0, resolution, 0.0, limit)
    raise ValueError(method)


def chamfer_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree
    if predicted.ndim != 2 or predicted.shape[1] != 3 or len(predicted) == 0:
        raise ValueError("Predicted point cloud must be a non-empty (N, 3) array.")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
        raise ValueError("Target point cloud must be a non-empty (N, 3) array.")
    pred_tree, target_tree = cKDTree(predicted), cKDTree(target)
    pred_dist = target_tree.query(predicted, k=1)[0]
    target_dist = pred_tree.query(target, k=1)[0]
    metrics: dict[str, float] = {
        "predicted_points": int(len(predicted)),
        "target_points": int(len(target)),
        "chamfer_l2": float((pred_dist ** 2).mean() + (target_dist ** 2).mean()),
        "mean_predicted_to_target": float(pred_dist.mean()),
        "mean_target_to_predicted": float(target_dist.mean()),
    }
    target_min, target_max = target.min(axis=0), target.max(axis=0)
    outside = np.any((predicted < target_min) | (predicted > target_max), axis=1)
    metrics["predicted_outside_target_bounds_fraction"] = float(outside.mean())
    metrics["predicted_inside_target_bounds_fraction"] = float((~outside).mean())
    for threshold_name, threshold in (("0.01", 0.01), ("0.02", 0.02)):
        precision = float((pred_dist <= threshold).mean())
        recall = float((target_dist <= threshold).mean())
        metrics[f"precision_at_{threshold_name}"] = precision
        metrics[f"recall_at_{threshold_name}"] = recall
        metrics[f"f1_at_{threshold_name}"] = float(
            2 * precision * recall / max(precision + recall, 1e-8)
        )
    return metrics


def align_to_target(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Put a reconstructed cloud in the benchmark's canonical scene frame.

    Camera-only reconstruction can leave a uniform scale and translation
    ambiguity even when its rendered views are correct. Preserve the learned
    orientation while matching the benchmark centroid and unit-sphere radius.
    """
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    predicted_low, predicted_high = np.quantile(predicted, (0.01, 0.99), axis=0)
    target_low, target_high = np.quantile(target, (0.01, 0.99), axis=0)
    predicted_extent = float(np.max(predicted_high - predicted_low))
    target_extent = float(np.max(target_high - target_low))
    if predicted_extent <= 1e-8 or target_extent <= 1e-8:
        raise ValueError("Cannot align a degenerate point cloud.")
    predicted_center = (predicted_low + predicted_high) * 0.5
    target_center = (target_low + target_high) * 0.5
    return (
        (predicted - predicted_center)
        * np.float32(target_extent / predicted_extent)
        + target_center
    )


def target_cloud(manifest: dict, object_name: str) -> np.ndarray:
    record = next(r for r in manifest["objects"] if f'{r["category"]}/{r["object_id"]}' == object_name)
    cloud = np.load(Path(manifest["dataset_dir"]) / record["point_cloud"], allow_pickle=False).astype(np.float32)
    normalization = record["normalization"]
    cloud = (cloud - np.asarray(normalization["center"], dtype=np.float32)) / float(normalization["scale"]) * 0.4
    # The benchmark files are stored in OmniObject3D's Y-up dataset frame;
    # convert them once to the scene frame used by all three trainers.
    return cloud[:, [0, 2, 1]]


def compare(method: str, predicted_path: Path, manifest_path: Path, object_name: str, output: Path) -> dict:
    predicted, colors = read_ply_data(predicted_path)
    target = target_cloud(json.loads(manifest_path.read_text()), object_name)
    predicted = align_to_target(predicted, target)
    write_ply(predicted_path, predicted, colors)
    metrics = chamfer_metrics(predicted, target)
    output.write_text(json.dumps(metrics, indent=2))
    return metrics


def read_ply(path: Path) -> np.ndarray:
    return read_ply_data(path)[0]


def read_ply_data(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    lines = path.read_text().splitlines()
    end = lines.index("end_header")
    count = next(int(line.split()[2]) for line in lines[:end] if line.startswith("element vertex"))
    rows = [line.split() for line in lines[end + 1:end + 1 + count]]
    points = np.asarray([[float(x) for x in row[:3]] for row in rows], dtype=np.float32)
    colors = None
    if rows and len(rows[0]) >= 6:
        colors = np.asarray([[int(float(x)) for x in row[3:6]] for row in rows], dtype=np.uint8)
    return points, colors
