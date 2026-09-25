#!/usr/bin/env python3
"""Multi-view NeRF oracle over OmniObject3D renders.

This is a *multi-view* upper-bound baseline, not a single-view generator: it
optimises a small NeRF per object from the 24 posed renders in
``dataset/renders/<cat>/<obj>/{00..23}.png`` + ``transforms.json`` and reports
held-out-view PSNR/SSIM.  It is deliberately dependency-light: a pure-PyTorch
MLP + volume renderer (no COLMAP, no CUDA extensions), so it runs on this box
where nvcc (12.8) does not match torch's CUDA (13.0).

See docs/ROADMAP.md for where results belong.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_object(dataset_dir: Path, category: str, object_id: str, res: int):
    """Return (images [V,S,S,3], poses [V,4,4], focal_px, name) composited on white."""
    obj_dir = dataset_dir / "renders" / category / object_id
    with open(obj_dir / "transforms.json") as f:
        tf = json.load(f)
    angle = tf["camera_angle_x"]
    orig = Image.open(obj_dir / tf["frames"][0]["file_path"]).size[0]
    focal = 0.5 * res / math.tan(0.5 * angle)

    imgs, poses = [], []
    for frame in sorted(tf["frames"], key=lambda f: f["file_path"]):
        im = Image.open(obj_dir / frame["file_path"]).convert("RGBA").resize((res, res), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        rgb = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4])  # white background
        imgs.append(rgb)
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
    return np.stack(imgs), np.stack(poses), focal, orig


def get_rays(c2w: torch.Tensor, focal: float, size: int, device):
    i, j = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=device),
        torch.arange(size, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [(i + 0.5 - size * 0.5) / focal, -(j + 0.5 - size * 0.5) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    rays_d = dirs @ c2w[:3, :3].t()
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


# ----------------------------------------------------------------------------
# NeRF
# ----------------------------------------------------------------------------
class Embedding(nn.Module):
    def __init__(self, in_dim, max_freq):
        super().__init__()
        self.freqs = 2.0 ** torch.arange(max_freq) * math.pi
        self.out_dim = in_dim * (1 + 2 * max_freq)

    def forward(self, x):
        xb = x[..., None, :] * self.freqs.to(x.device)[None, :, None]
        out = torch.cat([
            x,
            torch.sin(xb).reshape(*x.shape[:-1], -1),
            torch.cos(xb).reshape(*x.shape[:-1], -1),
        ], dim=-1)
        return out


class NeRF(nn.Module):
    def __init__(self, pos_freq=10, dir_freq=4, hidden=256, layers=8, skip=4):
        super().__init__()
        self.embed = Embedding(3, pos_freq)
        self.dir_embed = Embedding(3, dir_freq)
        self.skip = skip
        self.pts_linears = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.pts_linears.append(nn.Linear(self.embed.out_dim, hidden))
            elif i == skip:
                self.pts_linears.append(nn.Linear(hidden + self.embed.out_dim, hidden))
            else:
                self.pts_linears.append(nn.Linear(hidden, hidden))
        self.sigma = nn.Linear(hidden, 1)
        self.feature = nn.Linear(hidden, hidden)
        self.rgb = nn.Linear(hidden + self.dir_embed.out_dim, 3)

    def forward(self, x, d):
        h = self.embed(x)
        for i, layer in enumerate(self.pts_linears):
            if i == self.skip:
                h = torch.cat([h, self.embed(x)], dim=-1)
            h = F.relu(layer(h))
        sigma = F.relu(self.sigma(h))
        feat = self.feature(h)
        rgb = torch.sigmoid(self.rgb(torch.cat([feat, self.dir_embed(d)], dim=-1)))
        return sigma, rgb


def render(model, rays_o, rays_d, near, far, n_samples, rand=True, chunk=8192):
    outs = []
    for start in range(0, rays_o.shape[0], chunk):
        o, d = rays_o[start:start + chunk], rays_d[start:start + chunk]
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=o.device).expand(o.shape[0], n_samples)
        if rand:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_vals = lower + (upper - lower) * torch.rand_like(t_vals)
        z = near + (far - near) * t_vals
        pts = o[:, None, :] + d[:, None, :] * z[..., None]
        dirs = d[:, None, :].expand_as(pts)
        sigma, rgb = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
        sigma = sigma.reshape(o.shape[0], n_samples)
        rgb = rgb.reshape(o.shape[0], n_samples, 3)

        dists = torch.cat([z[:, 1:] - z[:, :-1], 1e-3 * torch.ones_like(z[:, :1])], dim=-1)
        alpha = 1.0 - torch.exp(-sigma * dists)
        trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        weights = alpha * trans
        color = (weights[..., None] * rgb).sum(dim=1) + (1.0 - weights.sum(dim=1, keepdim=True))
        outs.append(color)
    return torch.cat(outs, dim=0)


def ssim(a, b):
    """Simple global SSIM on [N,H,W,C] tensors in [0,1]."""
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)
    win = torch.ones(1, 1, 11, 11, device=a.device) / 121.0
    win = win.expand(a.shape[1], 1, 11, 11)
    mu_a = F.conv2d(a, win, padding=5, groups=a.shape[1])
    mu_b = F.conv2d(b, win, padding=5, groups=a.shape[1])
    var_a = F.conv2d(a * a, win, padding=5, groups=a.shape[1]) - mu_a ** 2
    var_b = F.conv2d(b * b, win, padding=5, groups=a.shape[1]) - mu_b ** 2
    cov = F.conv2d(a * b, win, padding=5, groups=a.shape[1]) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean().item()


def train_object(images, poses, focal, res, args, device):
    n_test = args.test_views
    order = np.arange(len(poses))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]
    images_t = torch.from_numpy(images).to(device)
    poses_t = torch.from_numpy(poses).to(device)

    near, far = args.near, args.far
    model = NeRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rays_cache = {}
    n_train = len(train_idx)
    for it in range(1, args.iters + 1):
        vi = train_idx[np.random.randint(n_train)]
        if vi not in rays_cache:
            rays_cache[vi] = get_rays(poses_t[vi], focal, res, device)
        o, d = rays_cache[vi]
        sel = torch.randint(0, o.shape[0], (args.batch,), device=device)
        pred = render(model, o[sel], d[sel], near, far, args.samples)
        loss = F.mse_loss(pred, images_t[vi].reshape(-1, 3)[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(1, args.iters // 5) == 0 or it == 1:
            print(f"    iter {it:5d}/{args.iters}  mse={loss.item():.5f}", flush=True)

    # evaluate held-out views
    psnrs, ssims = [], []
    with torch.no_grad():
        for vi in test_idx:
            o, d = get_rays(poses_t[vi], focal, res, device)
            pred = render(model, o, d, near, far, args.samples, rand=False).reshape(res, res, 3)
            gt = images_t[vi]
            mse = F.mse_loss(pred, gt).item()
            psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-10)))
            ssims.append(ssim(pred[None], gt[None]))
    return float(np.mean(psnrs)), float(np.mean(ssims)), pred, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "oracle")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="cat:object pairs; default = first object of a category sample")
    ap.add_argument("--res", type=int, default=128, help="render/train resolution")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=1024, help="rays per iteration")
    ap.add_argument("--samples", type=int, default=64, help="samples per ray")
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--far", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test-views", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    if args.objects:
        pairs = [tuple(o.split(":")) for o in args.objects]
    else:
        cats = ["medicine_bottle", "cup", "teapot", "hammer", "laptop", "tooth_paste"]
        pairs = []
        for c in cats:
            fs = sorted((args.dataset_dir / "point_clouds" / c).glob("*.npy"))
            if fs:
                pairs.append((c, fs[0].stem))

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cat, obj in pairs:
        name = f"{cat}_{obj}"
        print(f"[oracle] {name} ({args.res}px, {args.iters} iters)", flush=True)
        t0 = time.time()
        images, poses, focal, orig = load_object(args.dataset_dir, cat, obj, args.res)
        psnr, ssim_val, pred, gt = train_object(images, poses, focal, args.res, args, device)
        dt = time.time() - t0
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8)).save(out_dir / "render.png")
        Image.fromarray((gt.cpu().numpy() * 255).astype(np.uint8)).save(out_dir / "gt.png")
        (out_dir / "summary.json").write_text(json.dumps(
            {"object": name, "psnr": psnr, "ssim": ssim_val, "n_views": len(poses),
             "n_test_views": args.test_views, "res": args.res, "iters": args.iters,
             "seconds": round(dt, 1)}, indent=2) + "\n")
        all_rows.append({"object": name, "psnr": psnr, "ssim": ssim_val, "seconds": round(dt, 1)})
        print(f"  -> PSNR={psnr:.2f} dB  SSIM={ssim_val:.3f}  ({dt:.0f}s)", flush=True)

    summary = {
        "method": "NeRF (pure-PyTorch, multi-view oracle)",
        "n_objects": len(all_rows),
        "res": args.res,
        "iters": args.iters,
        "n_train_views": 24 - args.test_views,
        "n_test_views": args.test_views,
        "mean_psnr": float(np.mean([r["psnr"] for r in all_rows])) if all_rows else None,
        "mean_ssim": float(np.mean([r["ssim"] for r in all_rows])) if all_rows else None,
        "per_object": all_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))


if __name__ == "__main__":
    main()
