#!/usr/bin/env python3
"""Train a compact calibrated, image-only NeuS-style reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import NeuS


def record_for(path: Path, object_name: str) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text())
    for record in manifest["objects"]:
        if f'{record["category"]}/{record["object_id"]}' == object_name:
            return manifest, record
    raise ValueError(f"Object not found: {object_name}")


def cameras(render_dir: Path, names: list[str], size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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
    return images, images.amax(dim=-1) > 0.02


def make_rays(viewmats: torch.Tensor, intrinsic: torch.Tensor, camera_ids: torch.Tensor, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = torch.linalg.inv(viewmats[camera_ids])
    xy = torch.cat((pixels, torch.ones((len(pixels), 1), device=pixels.device)), dim=-1)
    directions = (torch.linalg.inv(intrinsic) @ xy.T).T
    directions = directions / directions.norm(dim=-1, keepdim=True)
    world = torch.bmm(inverse[:, :3, :3], directions.unsqueeze(-1)).squeeze(-1)
    world = world / world.norm(dim=-1, keepdim=True)
    return inverse[:, :3, 3], world


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, count: int) -> torch.Tensor:
    weights = weights.detach() + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat((torch.zeros_like(pdf[:, :1]), pdf.cumsum(dim=-1)), dim=-1)
    samples = torch.rand((*weights.shape[:-1], count), device=weights.device)
    indices = torch.searchsorted(cdf, samples, right=True).clamp(1, cdf.shape[-1] - 1)
    lower, upper = indices - 1, indices
    cdf0 = torch.gather(cdf, 1, lower)
    cdf1 = torch.gather(cdf, 1, upper)
    bins0 = torch.gather(bins, 1, lower)
    bins1 = torch.gather(bins, 1, upper)
    fraction = ((samples - cdf0) / (cdf1 - cdf0).clamp_min(1e-5)).clamp(0, 1)
    return bins0 + fraction * (bins1 - bins0)


def render(
    model: NeuS,
    origins: torch.Tensor,
    directions: torch.Tensor,
    coarse_samples: int,
    fine_samples: int,
    near: float,
    far: float,
    create_graph: bool,
    cos_anneal_ratio: float,
    distances_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = distances_override
    if distances is None:
        distances = torch.linspace(near, far, coarse_samples, device=origins.device).expand(len(origins), -1)
    points = origins[:, None, :] + directions[:, None, :] * distances[..., None]
    flat_points = points.reshape(-1, 3).requires_grad_(True)
    flat_directions = directions[:, None, :].expand_as(points).reshape(-1, 3)
    sdf, features = model.sdf_network(flat_points)
    gradients = torch.autograd.grad(
        sdf.sum(), flat_points, create_graph=create_graph, retain_graph=True, only_inputs=True
    )[0]
    normals = torch.nn.functional.normalize(gradients, dim=-1)
    colors = model.color_network(flat_points, normals, flat_directions, features)
    sample_count = distances.shape[1]
    sdf = sdf.reshape(len(origins), sample_count)
    colors = colors.reshape(len(origins), sample_count, 3)
    gradients = gradients.reshape(len(origins), sample_count, 3)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    true_cos = ((directions[:, None, :] * gradients[:, :-1]).sum(dim=-1))
    true_cos = -(torch.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 torch.relu(-true_cos) * cos_anneal_ratio)
    interval = distances[:, 1:] - distances[:, :-1]
    estimated_prev = mid_sdf - true_cos * interval * 0.5
    estimated_next = mid_sdf + true_cos * interval * 0.5
    inv_s = model.inv_s()
    prev_cdf = torch.sigmoid(estimated_prev * inv_s)
    next_cdf = torch.sigmoid(estimated_next * inv_s)
    alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0, 1)
    alpha = torch.cat((alpha, torch.zeros_like(alpha[:, :1])), dim=1)
    transmittance = torch.cumprod(
        torch.cat((torch.ones((len(origins), 1), device=origins.device), 1 - alpha + 1e-7), dim=1),
        dim=1,
    )[:, :-1]
    weights = alpha * transmittance
    rgb = (weights[..., None] * colors).sum(dim=1)
    if fine_samples <= 0:
        return rgb, weights, gradients, sdf
    mids = 0.5 * (distances[:, 1:] + distances[:, :-1])
    fine = sample_pdf(mids, weights[:, 1:-1], fine_samples)
    distances = torch.sort(torch.cat((distances, fine), dim=-1), dim=-1).values
    return render(
        model, origins, directions, distances.shape[1], 0, near, far,
        create_graph, cos_anneal_ratio, distances
    )


def save_views(model: NeuS, viewmats: torch.Tensor, intrinsic: torch.Tensor, names: list[str], size: int, samples: int, output: Path, chunk: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    y, x = torch.meshgrid(torch.arange(size, device=viewmats.device), torch.arange(size, device=viewmats.device), indexing="ij")
    pixels = torch.stack((x.flatten().float(), y.flatten().float()), dim=-1)
    for index, name in enumerate(names):
        ids = torch.full((len(pixels),), index, device=viewmats.device, dtype=torch.long)
        origins, directions = make_rays(viewmats, intrinsic, ids, pixels)
        parts = []
        for origin, direction in zip(origins.split(chunk), directions.split(chunk)):
            parts.append(render(model, origin, direction, samples, 0, 0.1, 4.0, False, 1.0)[0].detach())
        image = torch.cat(parts).reshape(size, size, 3).clamp(0, 1).cpu().numpy()
        Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--object", default="battery/battery_001")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rays-per-step", type=int, default=2048)
    parser.add_argument("--coarse-samples", type=int, default=64)
    parser.add_argument("--fine-samples", type=int, default=64)
    parser.add_argument("--render-chunk", type=int, default=2048)
    parser.add_argument("--foreground-probability", type=float, default=0.75)
    parser.add_argument("--silhouette-weight", type=float, default=0.1)
    parser.add_argument("--eikonal-weight", type=float, default=0.1)
    parser.add_argument("--eikonal-points", type=int, default=2048)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=4.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
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
    viewmats, intrinsic = cameras(render_dir, names, args.image_size, device)
    model = NeuS().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for iteration in range(1, args.iterations + 1):
        camera_ids = torch.randint(len(names), (args.rays_per_step,), device=device)
        foreground = torch.rand(args.rays_per_step, device=device) < args.foreground_probability
        flat_masks = masks[camera_ids].flatten(1)
        random_indices = torch.randint(args.image_size * args.image_size, (args.rays_per_step,), device=device)
        foreground_indices = torch.multinomial(flat_masks.float() + 1e-6, 1).squeeze(-1)
        chosen = torch.where(foreground, foreground_indices, random_indices)
        pixels = torch.stack((chosen % args.image_size, chosen // args.image_size), dim=-1).float()
        origins, directions = make_rays(viewmats, intrinsic, camera_ids, pixels)
        target = images[camera_ids, pixels[:, 1].long(), pixels[:, 0].long()]
        anneal = min(1.0, iteration / max(1, args.iterations // 2))
        prediction, weights, gradients, _ = render(
            model, origins, directions, args.coarse_samples, args.fine_samples,
            args.near, args.far, True, anneal
        )
        target_mask = masks[camera_ids, pixels[:, 1].long(), pixels[:, 0].long()].float()
        predicted_mask = weights.sum(dim=-1).clamp(0, 1)
        rgb_loss = torch.nn.functional.mse_loss(prediction, target)
        silhouette_loss = torch.nn.functional.binary_cross_entropy(predicted_mask.clamp(1e-5, 1 - 1e-5), target_mask)
        random_points = torch.rand(args.eikonal_points, 3, device=device) * 2.0 - 1.0
        random_points.requires_grad_(True)
        random_sdf, _ = model.sdf_network(random_points)
        random_gradients = torch.autograd.grad(random_sdf.sum(), random_points, create_graph=True)[0]
        eikonal_loss = ((gradients.norm(dim=-1) - 1.0) ** 2).mean()
        eikonal_loss = 0.5 * eikonal_loss + 0.5 * ((random_gradients.norm(dim=-1) - 1.0) ** 2).mean()
        loss = rgb_loss + args.silhouette_weight * silhouette_loss + args.eikonal_weight * eikonal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 100 == 0:
            print(f"iteration={iteration:05d} loss={loss.item():.6f}", flush=True)
        if iteration % args.checkpoint_every == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "image_size": args.image_size,
                        "coarse_samples": args.coarse_samples, "fine_samples": args.fine_samples,
                        "object": args.object, "iteration": iteration},
                       args.output_dir / f"checkpoint_{iteration:06d}.pt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "image_size": args.image_size,
                "coarse_samples": args.coarse_samples, "fine_samples": args.fine_samples,
                "object": args.object, "iteration": args.iterations}, args.output_dir / "checkpoint.pt")
    heldout = record["heldout_views"]
    heldout_viewmats, heldout_intrinsic = cameras(render_dir, heldout, args.image_size, device)
    save_views(model, heldout_viewmats, heldout_intrinsic, heldout, args.image_size,
               args.coarse_samples + args.fine_samples, args.output_dir / "heldout", args.render_chunk)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
