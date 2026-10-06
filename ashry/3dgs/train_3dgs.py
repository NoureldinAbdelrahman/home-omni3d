#!/usr/bin/env python3
"""Reconstruct one object from images using 3D Gaussian Splatting.

The reconstruction input is only the calibrated RGB images and
``transforms.json``. The benchmark point cloud is never loaded during
training; it is reserved for evaluation after reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from gsplat import DefaultStrategy, rasterization


def load_record(manifest_path: Path, object_name: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["objects"]:
        if f'{item["category"]}/{item["object_id"]}' == object_name:
            return manifest, item
    raise ValueError(f"Object not found in manifest: {object_name}")


def load_image(path: Path, size: int, device: torch.device) -> Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).to(device)


def load_cameras(
    render_dir: Path,
    names: list[str],
    size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    metadata = json.loads((render_dir / "transforms.json").read_text())
    by_name = {Path(frame["file_path"]).name: frame for frame in metadata["frames"]}
    focal = 0.5 * size / np.tan(float(metadata["camera_angle_x"]) * 0.5)
    Ks = torch.zeros((len(names), 3, 3), device=device)
    Ks[:, 0, 0] = float(focal)
    Ks[:, 1, 1] = float(focal)
    Ks[:, 0, 2] = size * 0.5
    Ks[:, 1, 2] = size * 0.5
    Ks[:, 2, 2] = 1.0

    viewmats = []
    for name in names:
        matrix = torch.tensor(by_name[name]["transform_matrix"], dtype=torch.float32)
        # transforms.json uses NeRF/OpenGL camera-to-world axes. gsplat uses
        # OpenCV-style camera axes and accepts world-to-camera matrices.
        matrix[:3, 1:3] *= -1
        viewmats.append(torch.linalg.inv(matrix))
    return torch.stack(viewmats).to(device), Ks


def save_render(path: Path, image: Tensor) -> None:
    array = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def initialize_colors(
    points: Tensor,
    images: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
) -> Tensor:
    """Average RGB samples from cameras that see each initialized point."""
    with torch.no_grad():
        n = points.shape[0]
        homogeneous = torch.cat(
            (points, torch.ones((n, 1), device=points.device)), dim=-1
        )
        camera_points = torch.einsum("cij,nj->cni", viewmats, homogeneous)[..., :3]
        projected = torch.einsum("cij,cnj->cni", Ks, camera_points)
        pixels = projected[..., :2] / camera_points[..., 2:].clamp_min(1e-6)
        height, width = images.shape[1:3]
        x = pixels[..., 0].round().long()
        y = pixels[..., 1].round().long()
        visible = (
            (camera_points[..., 2] > 0)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        colors = torch.full((n, 3), 0.5, device=points.device)
        sums = torch.zeros_like(colors)
        counts = torch.zeros((n, 1), device=points.device)
        for camera in range(len(images)):
            valid = visible[camera]
            if valid.any():
                samples = images[camera, y[camera, valid], x[camera, valid]]
                sums[valid] += samples
                counts[valid] += 1
        return torch.where(counts > 0, sums / counts.clamp_min(1), colors)


def initialize_visual_hull(
    images: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    count: int,
    candidates: int,
    min_views: int,
    seed: int,
) -> Tensor:
    """Sample 3D points that project inside foreground silhouettes."""
    generator = torch.Generator(device=images.device).manual_seed(seed)
    points = torch.rand(
        (candidates, 3), generator=generator, device=images.device
    ) * 0.9 - 0.45
    homogeneous = torch.cat(
        (points, torch.ones((candidates, 1), device=images.device)), dim=-1
    )
    camera_points = torch.einsum("cij,nj->cni", viewmats, homogeneous)[..., :3]
    projected = torch.einsum("cij,cnj->cni", Ks, camera_points)
    pixels = projected[..., :2] / camera_points[..., 2:].clamp_min(1e-6)
    height, width = images.shape[1:3]
    x = pixels[..., 0].round().long()
    y = pixels[..., 1].round().long()
    visible = (
        (camera_points[..., 2] > 0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    foreground = torch.zeros((len(images), candidates), dtype=torch.bool, device=images.device)
    for camera in range(len(images)):
        valid = visible[camera]
        if valid.any():
            foreground[camera, valid] = (
                images[camera, y[camera, valid], x[camera, valid]].max(dim=-1).values
                > 0.02
            )
    keep = foreground.sum(dim=0) >= min_views
    points = points[keep]
    if len(points) < count:
        raise RuntimeError(
            f"Visual-hull initialization retained {len(points)} points, "
            f"but {count} are required. Increase --visual-hull-candidates "
            "or lower --visual-hull-min-views."
        )
    selected = torch.randperm(len(points), generator=generator, device=images.device)[:count]
    return points[selected]


def initialize_log_scales(points: Tensor, minimum: float, maximum: float) -> Tensor:
    """Use local nearest-neighbor spacing for initial Gaussian sizes."""
    with torch.no_grad():
        distances = torch.cdist(points, points)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        return nearest.clamp(min=minimum, max=maximum).log().unsqueeze(-1).repeat(1, 3)


def render_views(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_size: int,
    output_dir: Path,
    names: list[str],
) -> None:
    """Render a camera batch and save files using the benchmark view names."""
    with torch.no_grad():
        rendered, _, _ = rasterization(
            means,
            F.normalize(quats, dim=-1),
            torch.exp(scales).clamp_min(1e-4),
            torch.sigmoid(opacities),
            colors,
            viewmats,
            Ks,
            image_size,
            image_size,
            backgrounds=torch.zeros((len(names), 3), device=means.device),
            packed=False,
            sh_degree=1,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, name in zip(rendered[..., :3], names):
        save_render(output_dir / name, image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="battery/battery_001")
    parser.add_argument(
        "--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--max-log-scale",
        type=float,
        default=-3.5,
        help="Maximum learned log standard deviation; lower values keep renders sharper.",
    )
    parser.add_argument(
        "--max-opacity-logit",
        type=float,
        default=2.0,
        help="Maximum opacity logit used during this conservative prototype run.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sh-degree", type=int, default=1, choices=(1,))
    parser.add_argument("--foreground-weight", type=float, default=0.1)
    parser.add_argument("--refine-start", type=int, default=500)
    parser.add_argument("--refine-stop", type=int, default=9000)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--init-points", type=int, default=4096)
    parser.add_argument("--visual-hull-candidates", type=int, default=150000)
    parser.add_argument("--visual-hull-min-views", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, record = load_record(args.manifest, args.object)
    dataset_dir = Path(manifest["dataset_dir"])
    render_dir = dataset_dir / record["render_dir"]
    input_names = record["input_views"]

    images = torch.stack(
        [load_image(render_dir / name, args.image_size, device) for name in input_names]
    )
    viewmats, Ks = load_cameras(
        render_dir, input_names, args.image_size, device
    )

    # Initialize from silhouettes only. The point cloud in the manifest is
    # intentionally not loaded here; it is ground truth for later evaluation.
    target = initialize_visual_hull(
        images,
        viewmats,
        Ks,
        args.init_points,
        args.visual_hull_candidates,
        args.visual_hull_min_views,
        args.seed,
    )
    n = target.shape[0]
    initial_colors = initialize_colors(target, images, viewmats, Ks)
    initial_scales = initialize_log_scales(target, 0.002, 0.03)
    means = torch.nn.Parameter(target.clone())
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    )
    scales = torch.nn.Parameter(initial_scales)
    opacities = torch.nn.Parameter(torch.full((n,), -4.0, device=device))
    # gsplat's SH convention uses a DC coefficient that is shifted by 0.5
    # and scaled by the degree-zero SH constant during rendering.
    sh = torch.zeros((n, 4, 3), device=device)
    sh[:, 0] = (initial_colors - 0.5) / 0.2820947917738781
    colors = torch.nn.Parameter(sh)
    params = torch.nn.ParameterDict(
        {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }
    )
    optimizers = {
        name: torch.optim.Adam([parameter], lr=lr, eps=1e-15)
        for name, parameter, lr in (
            ("means", means, 1.0e-5),
            ("quats", quats, 1.0e-4),
            ("scales", scales, 1.0e-3),
            ("opacities", opacities, 1.0e-2),
            ("colors", colors, 1.0e-2),
        )
    }
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        refine_start_iter=args.refine_start,
        refine_stop_iter=args.refine_stop,
        refine_every=args.refine_every,
        reset_every=3000,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    output_dir = args.output_dir / args.object.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.iterations + 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        indices = torch.randperm(len(images), device=device)[: min(4, len(images))]
        renders, alphas, info = rasterization(
            params["means"],
            F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]).clamp_min(1e-4),
            torch.sigmoid(params["opacities"]),
            params["colors"],
            viewmats[indices],
            Ks[indices],
            args.image_size,
            args.image_size,
            backgrounds=torch.zeros((len(indices), 3), device=device),
            packed=False,
            sh_degree=args.sh_degree,
            absgrad=True,
        )
        target_images = images[indices]
        foreground = (target_images.max(dim=-1).values > 0.02).float()
        loss_rgb = F.l1_loss(renders[..., :3], target_images)
        loss_fg = F.binary_cross_entropy(
            alphas[..., 0].clamp(1e-4, 1 - 1e-4), foreground
        )
        loss = loss_rgb + args.foreground_weight * loss_fg
        if (
            not torch.isfinite(renders).all()
            or not torch.isfinite(loss)
            or float(renders.detach().abs().max()) > 10.0
        ):
            raise RuntimeError(
                f"Invalid 3DGS render at iteration {iteration}; "
                "check camera conventions and scene normalization."
            )
        strategy.step_pre_backward(
            params, optimizers, strategy_state, iteration, info
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=1.0)
        strategy.step_post_backward(
            params,
            optimizers,
            strategy_state,
            iteration,
            info,
            packed=False,
        )
        for optimizer in optimizers.values():
            optimizer.step()
        with torch.no_grad():
            params["means"].clamp_(-1.0, 1.0)
            params["scales"].clamp_(-8.0, args.max_log_scale)
            params["opacities"].clamp_(-8.0, args.max_opacity_logit)
            params["colors"].clamp_(-8.0, 8.0)

        if iteration == 1 or iteration % 100 == 0:
            print(f"iteration={iteration:05d} loss={loss.item():.6f}")
        if iteration % args.save_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                preview, _, _ = rasterization(
                    params["means"],
                    F.normalize(params["quats"], dim=-1),
                    torch.exp(params["scales"]).clamp_min(1e-4),
                    torch.sigmoid(params["opacities"]),
                    params["colors"],
                    viewmats[:1],
                    Ks[:1],
                    args.image_size,
                    args.image_size,
                    backgrounds=torch.zeros((1, 3), device=device),
                    packed=False,
                    sh_degree=args.sh_degree,
                )
            save_render(output_dir / f"train_{iteration:05d}.png", preview[0, ..., :3])

    with torch.no_grad():
        weights = torch.sigmoid(params["opacities"])
        keep = weights > 0.05
        np.save(
            output_dir / "gaussian_centers.npy",
            params["means"].detach()[keep].cpu().numpy().astype(np.float32),
        )
    torch.save(
        {
            "means": params["means"].detach().cpu(),
            "quats": params["quats"].detach().cpu(),
            "scales": params["scales"].detach().cpu(),
            "opacities": params["opacities"].detach().cpu(),
            "colors": params["colors"].detach().cpu(),
            "record": record,
        },
        output_dir / "checkpoint.pt",
    )
    heldout_names = record["heldout_views"]
    heldout_viewmats, heldout_Ks = load_cameras(
        render_dir, heldout_names, args.image_size, device
    )
    render_views(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["colors"],
        heldout_viewmats,
        heldout_Ks,
        args.image_size,
        output_dir / "heldout",
        heldout_names,
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
