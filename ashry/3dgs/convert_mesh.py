#!/usr/bin/env python3
"""Convert a trained 3DGS checkpoint into an approximate density mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes


def quat_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True).clip(1e-8)
    w, x, y, z = quaternions.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def density_grid(checkpoint: dict, resolution: int, margin: float, sigma_extent: float) -> tuple[np.ndarray, np.ndarray]:
    means = checkpoint["means"].numpy()
    scales = np.exp(checkpoint["scales"].numpy()).clip(1e-4, 1.0)
    rotations = quat_to_matrix(checkpoint["quats"].numpy())
    opacities = 1.0 / (1.0 + np.exp(-checkpoint["opacities"].numpy()))
    bounds_min = (means - sigma_extent * scales.max(axis=1, keepdims=True)).min(axis=0) - margin
    bounds_max = (means + sigma_extent * scales.max(axis=1, keepdims=True)).max(axis=0) + margin
    extent = bounds_max - bounds_min
    bounds_min = (bounds_min + bounds_max) * 0.5 - max(extent) * 0.5
    bounds_max = (bounds_min + max(extent)).astype(np.float32)
    axis = [
        np.linspace(bounds_min[index], bounds_max[index], resolution, dtype=np.float32)
        for index in range(3)
    ]
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for mean, scale, rotation, opacity in zip(means, scales, rotations, opacities):
        radius = sigma_extent * scale
        lower = np.maximum(np.floor((mean - radius - bounds_min) / (bounds_max - bounds_min) * (resolution - 1)).astype(int), 0)
        upper = np.minimum(np.ceil((mean + radius - bounds_min) / (bounds_max - bounds_min) * (resolution - 1)).astype(int) + 1, resolution)
        if np.any(upper <= lower) or opacity < 1e-4:
            continue
        xs, ys, zs = np.meshgrid(
            axis[0][lower[0]:upper[0]],
            axis[1][lower[1]:upper[1]],
            axis[2][lower[2]:upper[2]],
            indexing="ij",
        )
        local = np.stack((xs - mean[0], ys - mean[1], zs - mean[2]), axis=-1)
        local = local @ rotation
        value = opacity * np.exp(-0.5 * np.sum((local / scale) ** 2, axis=-1))
        density[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] += value.astype(np.float32)
    return density, bounds_min


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--level", type=float, default=0.08)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--sigma-extent", type=float, default=3.0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    density, bounds_min = density_grid(checkpoint, args.resolution, args.margin, args.sigma_extent)
    if density.max() <= args.level:
        raise RuntimeError(f"Density maximum {density.max():.6f} is below level {args.level}")
    vertices, faces, _, _ = marching_cubes(density, level=args.level)
    scale = (max(density.shape) - 1)
    means = checkpoint["means"].numpy()
    scales = np.exp(checkpoint["scales"].numpy())
    bounds_max = (means + args.sigma_extent * scales.max(axis=1, keepdims=True)).max(axis=0) + args.margin
    extent = max(bounds_max - bounds_min)
    vertices = bounds_min + vertices * (extent / scale)
    write_obj(args.output, vertices, faces)
    print(f"Wrote {args.output} ({len(vertices)} vertices, {len(faces)} faces, max_density={density.max():.4f})")


if __name__ == "__main__":
    main()
