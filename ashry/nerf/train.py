#!/usr/bin/env python3
"""Train a compact calibrated multi-view NeRF using RGB renders only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import NeRF


def record_for(manifest_path: Path, object_name: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["objects"]:
        if f'{record["category"]}/{record["object_id"]}' == object_name:
            return manifest, record
    raise ValueError(f"Object not found: {object_name}")


def load_cameras(render_dir: Path, names: list[str], size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = json.loads((render_dir / "transforms.json").read_text())
    frames = {Path(frame["file_path"]).name: frame for frame in metadata["frames"]}
    focal = 0.5 * size / np.tan(float(metadata["camera_angle_x"]) * 0.5)
    intrinsic = torch.tensor([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], device=device, dtype=torch.float32)
    poses = []
    for name in names:
        pose = torch.tensor(frames[name]["transform_matrix"], device=device, dtype=torch.float32)
        pose[:3, 1:3] *= -1
        poses.append(torch.linalg.inv(pose))
    return torch.stack(poses), intrinsic


def load_images(render_dir: Path, names: list[str], size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = []
    for name in names:
        image = Image.open(render_dir / name).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    images = torch.from_numpy(np.stack(arrays)).to(device)
    masks = images.amax(dim=-1) > 0.02
    return images, masks


def make_rays(viewmats: torch.Tensor, intrinsic: torch.Tensor, camera_ids: torch.Tensor, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = torch.linalg.inv(viewmats[camera_ids])
    xy = torch.cat((pixels, torch.ones((len(pixels), 1), device=pixels.device)), dim=-1)
    directions = (torch.linalg.inv(intrinsic) @ xy.T).T
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    world_directions = torch.bmm(inverse[:, :3, :3], directions.unsqueeze(-1)).squeeze(-1)
    world_directions = world_directions / torch.linalg.norm(world_directions, dim=-1, keepdim=True)
    origins = inverse[:, :3, 3]
    return origins, world_directions


def raw_render(model: NeRF, origins: torch.Tensor, directions: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = origins[:, None, :] + directions[:, None, :] * distances[..., None]
    flat_points = points.reshape(-1, 3)
    flat_directions = directions[:, None, :].expand_as(points).reshape(-1, 3)
    density, color = model(flat_points, flat_directions)
    sample_count = distances.shape[1]
    density = density.reshape(len(origins), sample_count)
    color = color.reshape(len(origins), sample_count, 3)
    deltas = distances[:, 1:] - distances[:, :-1]
    deltas = torch.cat((deltas, torch.full_like(deltas[:, :1], 1e10)), dim=1)
    alpha = 1.0 - torch.exp(-density * deltas)
    transmittance = torch.cumprod(torch.cat((torch.ones((len(origins), 1), device=origins.device), 1 - alpha + 1e-7), dim=1), dim=1)[:, :-1]
    weights = alpha * transmittance
    return (weights[..., None] * color).sum(dim=1), weights


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, count: int) -> torch.Tensor:
    weights = weights.detach() + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat((torch.zeros_like(pdf[:, :1]), pdf.cumsum(dim=-1)), dim=-1)
    samples = torch.rand((*weights.shape[:-1], count), device=weights.device)
    indices = torch.searchsorted(cdf, samples, right=True).clamp(1, cdf.shape[-1] - 1)
    lower = (indices - 1).clamp_min(0)
    upper = indices
    cdf_lower = torch.gather(cdf, 1, lower)
    cdf_upper = torch.gather(cdf, 1, upper)
    bins_lower = torch.gather(bins, 1, lower)
    bins_upper = torch.gather(bins, 1, upper)
    fraction = ((samples - cdf_lower) / (cdf_upper - cdf_lower).clamp_min(1e-5)).clamp(0, 1)
    return bins_lower + fraction * (bins_upper - bins_lower)


def render(
    model: NeRF,
    origins: torch.Tensor,
    directions: torch.Tensor,
    coarse_samples: int,
    fine_samples: int,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_distances = torch.linspace(near, far, coarse_samples, device=origins.device)
    coarse_distances = coarse_distances.expand(len(origins), -1)
    coarse_rgb, coarse_weights = raw_render(model, origins, directions, coarse_distances)
    mids = 0.5 * (coarse_distances[:, 1:] + coarse_distances[:, :-1])
    fine_distances = sample_pdf(mids, coarse_weights[:, 1:-1], fine_samples)
    distances = torch.sort(torch.cat((coarse_distances, fine_distances), dim=-1), dim=-1).values
    return raw_render(model, origins, directions, distances)


def save_views(model: NeRF, viewmats: torch.Tensor, intrinsic: torch.Tensor, names: list[str], size: int, coarse_samples: int, fine_samples: int, output: Path, chunk: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    y, x = torch.meshgrid(torch.arange(size, device=viewmats.device), torch.arange(size, device=viewmats.device), indexing="ij")
    pixels = torch.stack((x.flatten().float(), y.flatten().float()), dim=-1)
    with torch.no_grad():
        for index, name in enumerate(names):
            camera_ids = torch.full((len(pixels),), index, device=viewmats.device, dtype=torch.long)
            origins, directions = make_rays(viewmats, intrinsic, camera_ids, pixels)
            chunks = [render(model, o, d, coarse_samples, fine_samples, 0.1, 4.0)[0] for o, d in zip(origins.split(chunk), directions.split(chunk))]
            image = torch.cat(chunks).reshape(size, size, 3).clamp(0, 1).cpu().numpy()
            Image.fromarray((image * 255).astype(np.uint8)).save(output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--object", default="battery/battery_001")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--rays-per-step", type=int, default=4096)
    parser.add_argument("--coarse-samples", type=int, default=64)
    parser.add_argument("--fine-samples", type=int, default=64)
    parser.add_argument("--render-chunk", type=int, default=4096)
    parser.add_argument("--foreground-probability", type=float, default=0.75)
    parser.add_argument("--silhouette-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/battery_battery_001"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, record = record_for(args.manifest, args.object)
    render_dir = Path(manifest["dataset_dir"]) / record["render_dir"]
    names = record["input_views"]
    images, masks = load_images(render_dir, names, args.image_size, device)
    viewmats, intrinsic = load_cameras(render_dir, names, args.image_size, device)
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for iteration in range(1, args.iterations + 1):
        foreground = torch.rand(args.rays_per_step, device=device) < args.foreground_probability
        camera_ids = torch.randint(len(names), (args.rays_per_step,), device=device)
        flat_masks = masks[camera_ids].reshape(args.rays_per_step, -1)
        flat_indices = torch.randint(args.image_size * args.image_size, (args.rays_per_step,), device=device)
        foreground_indices = torch.multinomial(flat_masks.float() + 1e-6, 1).squeeze(-1)
        chosen = torch.where(foreground, foreground_indices, flat_indices)
        pixels = torch.stack((chosen % args.image_size, chosen // args.image_size), dim=-1).float()
        origins, directions = make_rays(viewmats, intrinsic, camera_ids, pixels)
        target = images[camera_ids, pixels[:, 1].long().clamp_max(args.image_size - 1), pixels[:, 0].long().clamp_max(args.image_size - 1)]
        prediction, weights = render(model, origins, directions, args.coarse_samples, args.fine_samples, 0.1, 4.0)
        target_mask = masks[camera_ids, pixels[:, 1].long(), pixels[:, 0].long()].float()
        predicted_mask = weights.sum(dim=-1).clamp(0, 1)
        rgb_loss = torch.nn.functional.mse_loss(prediction, target)
        silhouette_loss = torch.nn.functional.binary_cross_entropy(predicted_mask.clamp(1e-5, 1 - 1e-5), target_mask)
        loss = rgb_loss + args.silhouette_weight * silhouette_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0:
            print(f"iteration={iteration:05d} loss={loss.item():.6f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "image_size": args.image_size, "coarse_samples": args.coarse_samples, "fine_samples": args.fine_samples, "object": args.object}, args.output_dir / "checkpoint.pt")
    heldout = record["heldout_views"]
    heldout_viewmats, heldout_intrinsic = load_cameras(render_dir, heldout, args.image_size, device)
    save_views(model, heldout_viewmats, heldout_intrinsic, heldout, args.image_size, args.coarse_samples, args.fine_samples, args.output_dir / "heldout", args.render_chunk)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
