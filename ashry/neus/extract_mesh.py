#!/usr/bin/env python3
"""Extract an OBJ surface from a trained NeuS SDF checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes

from model import NeuS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--min-bound", type=float, nargs=3, default=(-1.0, -1.0, -1.0))
    parser.add_argument("--max-bound", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--chunk", type=int, default=65536)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuS()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    lower = np.asarray(args.min_bound, dtype=np.float32)
    upper = np.asarray(args.max_bound, dtype=np.float32)
    axes = [torch.linspace(float(lower[i]), float(upper[i]), args.resolution, device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    values = []
    for points in grid.split(args.chunk):
        with torch.no_grad():
            sdf, _ = model.sdf_network(points)
        values.append(sdf[:, 0].cpu())
    sdf = torch.cat(values).reshape(args.resolution, args.resolution, args.resolution).numpy()
    if sdf.min() > 0 or sdf.max() < 0:
        raise RuntimeError(f"SDF grid does not cross zero: min={sdf.min():.5f}, max={sdf.max():.5f}")
    vertices, faces, _, _ = marching_cubes(sdf, level=0.0)
    vertices = lower + vertices * ((upper - lower) / (args.resolution - 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"Wrote {args.output} ({len(vertices)} vertices, {len(faces)} faces)")


if __name__ == "__main__":
    main()
