#!/usr/bin/env python3
"""Extract NeRF meshes and compare them with the corresponding 3DGS meshes.

The textured OBJ exports use a tiled atlas of the original calibrated RGB views.
They are preferred over the vertex-colored PLY files when sharp image details
are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

NERF_DIR = Path(__file__).resolve().parents[1] / "nerf"
sys.path.insert(0, str(NERF_DIR))
from model import NeRF  # noqa: E402


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields and fields[0] == "f" and len(fields) >= 4:
            faces.append([int(field.split("/")[0]) - 1 for field in fields[1:4]])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no drawable vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} {color[0]} {color[1]} {color[2]}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_textured_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    render_dir: Path,
    names: list[str],
    size: int,
) -> None:
    """Write an OBJ whose UV atlas is a tiled copy of the source RGB views."""
    viewmats, images, focal = load_images_and_cameras(render_dir, names, size)
    columns = 5
    rows = int(np.ceil(len(names) / columns))
    atlas = Image.new("RGB", (columns * size, rows * size), (0, 0, 0))
    for index, image in enumerate(images):
        tile = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        atlas.paste(tile, ((index % columns) * size, (index // columns) * size))
    texture_path = path.with_name(f"{path.stem}_texture.png")
    atlas.save(texture_path)
    material_path = path.with_suffix(".mtl")
    material_path.write_text(
        f"newmtl source_views\nKd 1.0 1.0 1.0\nmap_Kd {texture_path.name}\n"
    )

    points = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float32)), axis=1)
    normals = np.zeros_like(vertices)
    for face in faces:
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        normals[face] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    camera_positions = np.linalg.inv(viewmats)[:, :3, 3]
    output_vertices, output_uvs, output_faces = [], [], []
    for face in faces:
        center = vertices[face].mean(axis=0)
        to_camera = camera_positions - center
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-8)
        scores = to_camera @ normals[face].mean(axis=0)
        candidates = []
        for camera_index, viewmat in enumerate(viewmats):
            projected = (viewmat @ points[face].T).T[:, :3]
            if np.any(projected[:, 2] <= 1e-5):
                continue
            pixels = np.column_stack(
                (focal * projected[:, 0] / projected[:, 2] + size * 0.5,
                 focal * projected[:, 1] / projected[:, 2] + size * 0.5)
            )
            if np.all((pixels >= 1) & (pixels <= size - 2)):
                candidates.append((float(scores[camera_index]), camera_index, projected, pixels))
        if candidates:
            _, best, projected, pixels = max(candidates, key=lambda item: item[0])
        else:
            best = int(np.argmax(scores))
            projected = (viewmats[best] @ points[face].T).T[:, :3]
            pixels = np.column_stack(
                (focal * projected[:, 0] / np.maximum(projected[:, 2], 1e-5) + size * 0.5,
                 focal * projected[:, 1] / np.maximum(projected[:, 2], 1e-5) + size * 0.5)
            )
            pixels = np.clip(pixels, 1, size - 2)
        tile_x, tile_y = best % columns, best // columns
        for vertex, pixel in zip(vertices[face], pixels):
            output_vertices.append(vertex)
            output_uvs.append(
                ((tile_x * size + pixel[0]) / (columns * size),
                 1.0 - (tile_y * size + pixel[1]) / (rows * size))
            )
        start = len(output_vertices) - 3
        output_faces.append((start + 1, start + 2, start + 3))
    with path.open("w") as handle:
        handle.write(f"mtllib {material_path.name}\nusemtl source_views\n")
        for vertex in output_vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in output_uvs:
            handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for face in output_faces:
            handle.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")


def load_images_and_cameras(
    render_dir: Path, names: list[str], size: int
) -> tuple[np.ndarray, np.ndarray, float]:
    from PIL import Image

    metadata = json.loads((render_dir / "transforms.json").read_text())
    frames = {Path(frame["file_path"]).name: frame for frame in metadata["frames"]}
    focal = 0.5 * size / np.tan(float(metadata["camera_angle_x"]) * 0.5)
    viewmats, images = [], []
    for name in names:
        pose = torch.tensor(frames[name]["transform_matrix"], dtype=torch.float32)
        pose[:3, 1:3] *= -1
        viewmats.append(torch.linalg.inv(pose).numpy())
        image = Image.open(render_dir / name).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.asarray(viewmats), np.asarray(images), focal


def bake_vertex_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    render_dir: Path,
    names: list[str],
    size: int,
) -> np.ndarray:
    viewmats, images, focal = load_images_and_cameras(render_dir, names, size)
    points = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float32)), axis=1)
    normals = np.zeros_like(vertices)
    for face in faces:
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        normals[face] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    colors = np.zeros((len(vertices), 3), dtype=np.float32)
    quality = np.full(len(vertices), -np.inf, dtype=np.float32)
    for viewmat, image in zip(viewmats, images):
        camera = (viewmat @ points.T).T[:, :3]
        valid = camera[:, 2] > 1e-5
        u = np.rint(focal * camera[:, 0] / camera[:, 2] + size * 0.5).astype(np.int32)
        v = np.rint(focal * camera[:, 1] / camera[:, 2] + size * 0.5).astype(np.int32)
        valid &= (u >= 0) & (u < size) & (v >= 0) & (v < size)
        camera_positions = np.linalg.inv(viewmat)[:3, 3]
        to_camera = camera_positions[None, :] - vertices
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-8)
        score = np.sum(normals * to_camera, axis=1)
        valid &= score > 0
        indices = np.flatnonzero(valid & (score > quality))
        colors[indices] = image[v[indices], u[indices]]
        quality[indices] = score[indices]
    fallback = colors[quality > -np.inf].mean(axis=0) if np.any(quality > -np.inf) else np.array([0.6, 0.6, 0.6], dtype=np.float32)
    colors[quality == -np.inf] = fallback
    return np.rint(colors.clip(0, 1) * 255).astype(np.uint8)


def nerf_mesh(
    checkpoint_path: Path,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    resolution: int,
    level: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = NeRF()
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    axis = [
        torch.linspace(float(bounds_min[i]), float(bounds_max[i]), resolution, device=device)
        for i in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axis, indexing="ij"), dim=-1).reshape(-1, 3)
    density_parts = []
    directions = torch.zeros_like(grid)
    with torch.no_grad():
        for points in grid.split(chunk):
            density, _ = model(points, directions[: len(points)])
            density_parts.append(density.cpu())
    density = torch.cat(density_parts).reshape(resolution, resolution, resolution).numpy()
    maximum = float(density.max())
    if maximum <= level:
        raise RuntimeError(
            f"NeRF density maximum {maximum:.4f} is below level {level}; "
            "try a lower --nerf-level or wider bounds."
        )
    occupied = density >= level
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        raise RuntimeError("No connected NeRF density component found at the requested level")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest = component_sizes.argmax()
    density[labels != largest] = 0.0
    vertices, faces, _, _ = marching_cubes(density, level=level)
    scale = (bounds_max - bounds_min) / (resolution - 1)
    vertices = bounds_min + vertices * scale
    return vertices.astype(np.float32), faces.astype(np.int64), maximum


def mesh_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    first_extent = first.max(axis=0) - first.min(axis=0)
    second_extent = second.max(axis=0) - second.min(axis=0)
    return {
        "chamfer_mean": float((first_to_second.mean() + second_to_first.mean()) * 0.5),
        "nerf_to_3dgs_mean": float(first_to_second.mean()),
        "3dgs_to_nerf_mean": float(second_to_first.mean()),
        "nerf_bbox_x": float(first_extent[0]),
        "nerf_bbox_y": float(first_extent[1]),
        "nerf_bbox_z": float(first_extent[2]),
        "3dgs_bbox_x": float(second_extent[0]),
        "3dgs_bbox_y": float(second_extent[1]),
        "3dgs_bbox_z": float(second_extent[2]),
    }


def panel(
    nerf_vertices: np.ndarray,
    nerf_faces: np.ndarray,
    gs_vertices: np.ndarray,
    gs_faces: np.ndarray,
    output: Path,
    views: int,
    size: int,
) -> None:
    all_vertices = np.concatenate((nerf_vertices, gs_vertices))
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) * 0.5
    extent = float(np.max(all_vertices.max(axis=0) - all_vertices.min(axis=0))) * 0.6
    output.mkdir(parents=True, exist_ok=True)
    for index in range(views):
        figure = plt.figure(figsize=(2 * size / 100, size / 100), dpi=100)
        for position, (vertices, faces, title, color) in enumerate(
            (
                (nerf_vertices, nerf_faces, "NeRF mesh", "#d88745"),
                (gs_vertices, gs_faces, "3DGS mesh", "#5aa6d6"),
            ),
            start=1,
        ):
            axis = figure.add_subplot(1, 2, position, projection="3d")
            polygons = (vertices - center)[faces]
            axis.add_collection3d(
                Poly3DCollection(polygons, facecolor=color, edgecolor="#333333", linewidth=0.08)
            )
            axis.set_xlim(-extent, extent)
            axis.set_ylim(-extent, extent)
            axis.set_zlim(-extent, extent)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=18, azim=360.0 * index / views)
            axis.set_title(title)
            axis.set_axis_off()
        figure.subplots_adjust(0, 0, 1, 0.92, wspace=0)
        figure.savefig(output / f"{index:03d}.png", facecolor="white", pad_inches=0)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerf-results", type=Path, default=Path("../nerf/results/category_sample_nerf_fast"))
    parser.add_argument("--gs-results", type=Path, default=Path("../3dgs/results/category_sample"))
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("qualitative/category_sample_nerf_3dgs_meshes"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--nerf-level", type=float, default=30.0)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--grid-chunk", type=int, default=65536)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--object", action="append", dest="objects",
                        help="Only process this result directory name; repeat for multiple objects")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    record_by_directory = {
        f'{record["category"]}_{record["object_id"]}': record
        for record in manifest["objects"]
    }
    rows = []
    for gs_dir in sorted(path for path in args.gs_results.iterdir() if (path / "3dgs_mesh.obj").is_file()):
        if args.objects and gs_dir.name not in args.objects:
            continue
        nerf_dir = args.nerf_results / gs_dir.name
        nerf_checkpoint = nerf_dir / "checkpoint.pt"
        if not nerf_checkpoint.is_file() and args.nerf_results.name == "battery_upgraded":
            nerf_checkpoint = args.nerf_results / "checkpoint.pt"
        if not nerf_checkpoint.is_file():
            print(f"Skipping {gs_dir.name}: NeRF checkpoint not found", flush=True)
            continue
        gs_vertices, gs_faces = load_obj(gs_dir / "3dgs_mesh.obj")
        lower = gs_vertices.min(axis=0)
        upper = gs_vertices.max(axis=0)
        extent = upper - lower
        lower = lower - np.maximum(extent * 0.35, 0.02)
        upper = upper + np.maximum(extent * 0.35, 0.02)
        object_dir = args.output_dir / gs_dir.name
        nerf_vertices, nerf_faces, maximum = nerf_mesh(
            nerf_checkpoint, lower, upper, args.resolution, args.nerf_level, args.grid_chunk
        )
        nerf_mesh_path = object_dir / "nerf_mesh.obj"
        gs_mesh_path = object_dir / "3dgs_mesh.obj"
        write_obj(nerf_mesh_path, nerf_vertices, nerf_faces)
        record = record_by_directory.get(gs_dir.name)
        if record is None:
            raise ValueError(f"Could not map result directory to manifest object: {gs_dir.name}")
        render_dir = Path(manifest["dataset_dir"]) / record["render_dir"]
        colors = bake_vertex_colors(nerf_vertices, nerf_faces, render_dir, record["input_views"], args.texture_size)
        write_ply(object_dir / "nerf_mesh_colored.ply", nerf_vertices, nerf_faces, colors)
        gs_colors = bake_vertex_colors(gs_vertices, gs_faces, render_dir, record["input_views"], args.texture_size)
        write_ply(object_dir / "3dgs_mesh_colored.ply", gs_vertices, gs_faces, gs_colors)
        write_textured_obj(
            object_dir / "nerf_mesh_textured.obj",
            nerf_vertices,
            nerf_faces,
            render_dir,
            record["input_views"],
            args.texture_size,
        )
        write_textured_obj(
            object_dir / "3dgs_mesh_textured.obj",
            gs_vertices,
            gs_faces,
            render_dir,
            record["input_views"],
            args.texture_size,
        )
        write_obj(gs_mesh_path, gs_vertices, gs_faces)
        metrics = mesh_metrics(nerf_vertices, gs_vertices)
        metrics.update({"object": gs_dir.name, "nerf_density_max": maximum})
        rows.append(metrics)
        panel(nerf_vertices, nerf_faces, gs_vertices, gs_faces, object_dir / "turntable", args.views, args.size)
        print(f"Processed {gs_dir.name}", flush=True)
    fields = ["object", "chamfer_mean", "nerf_to_3dgs_mean", "3dgs_to_nerf_mean",
              "nerf_bbox_x", "nerf_bbox_y", "nerf_bbox_z", "3dgs_bbox_x",
              "3dgs_bbox_y", "3dgs_bbox_z", "nerf_density_max"]
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"Processed {len(rows)} objects; wrote {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
