#!/usr/bin/env python3
"""Render an OBJ mesh from a set of turntable viewpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            face = []
            for field in fields[1:4]:
                face.append(int(field.split("/")[0]) - 1)
            faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no drawable vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def render(vertices: np.ndarray, faces: np.ndarray, output: Path, views: int, size: int) -> None:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0))) * 0.6
    normalized = vertices - center
    output.mkdir(parents=True, exist_ok=True)
    for index in range(views):
        azimuth = 360.0 * index / views
        figure = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        axis = figure.add_subplot(111, projection="3d")
        polygons = normalized[faces]
        axis.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolor="#5aa6d6",
                edgecolor="#174a68",
                linewidth=0.08,
                alpha=1.0,
            )
        )
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_zlim(-extent, extent)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=18, azim=azimuth)
        axis.set_axis_off()
        figure.subplots_adjust(0, 0, 1, 1)
        figure.savefig(output / f"{index:03d}.png", facecolor="white", pad_inches=0)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    vertices, faces = load_obj(args.mesh)
    render(vertices, faces, args.output_dir, args.views, args.size)
    print(f"Wrote {args.views} mesh renders to {args.output_dir}")


if __name__ == "__main__":
    main()
