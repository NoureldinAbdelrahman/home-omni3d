"""Shared plotting helpers for the analysis notebook.

Keeps matplotlib safe in both Jupyter and headless execution: figures are always
saved to disk, and ``show`` only calls ``plt.show`` when an interactive backend
is active (so the ``FigureCanvasAgg is non-interactive`` warning never appears).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

# Always render to the Agg canvas: figures are embedded with show_saved() or
# shown by the notebook's own inline backend. Forcing TkAgg here would make
# headless execution (nbconvert) hang trying to reach an X display.
import os

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_INTERACTIVE = False


def show(fig=None):
    """Show a figure only when the backend is interactive (no warning)."""
    if _INTERACTIVE:
        plt.show()
    if fig is not None:
        plt.close(fig)


def show_saved(path, width: int = 820):
    """Embed a saved PNG as a cell output, so figures show inline even when the
    notebook was executed headless (Agg), where ``plt.show()`` renders nothing."""
    from IPython.display import Image, display

    path = Path(path)
    if not Path(path).is_file():
        print(f"[plotting] figure missing: {path}")
        return
    try:
        display(Image(filename=str(path), width=width))
    except Exception as exc:  # display not available (e.g. plain script)
        print(f"[plotting] could not embed {path.name}: {exc}")
    plt.close("all")


def savefig(fig, path: Path, root: Path | None = None, dpi: int = 150) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    label = path.relative_to(root) if root is not None else path
    return f"saved {label}"


def normalize_cloud(pts: np.ndarray) -> np.ndarray:
    """Center and scale a point cloud to the unit sphere (robust to outliers)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return pts
    center = np.median(pts, axis=0)
    pts = pts - center
    # Use a high percentile radius so a few strays don't shrink the cloud.
    radius = np.percentile(np.linalg.norm(pts, axis=1), 99)
    if radius < 1e-9:
        radius = np.linalg.norm(pts, axis=1).max()
    if radius < 1e-9:
        return pts
    return pts / radius


def voxel_to_rgb(pred: np.ndarray, gt: np.ndarray, thr: float = 0.3):
    """Max-projection RGB of pred (red) vs gt (teal): overlap looks gray/white."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]
    p = pred.max(axis=2).T
    g = gt.max(axis=2).T
    p = (p >= thr).astype(float) if pred.max() > 1.0 else np.clip(p, 0, 1)
    g = (g >= 0.5).astype(float)
    rgb = np.zeros(p.shape + (3,), dtype=float)
    rgb[..., 0] = p          # red  = predicted
    rgb[..., 1] = g          # green+blue = teal = expected
    rgb[..., 2] = g
    return rgb


def voxel_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a readable 2D slice through the middle of a volume."""
    vol = np.asarray(vol, dtype=float)
    if vol.ndim == 4:
        vol = vol[0]
    if axis == 0:
        sl = vol[index, :, :].T
    elif axis == 1:
        sl = vol[:, index, :].T
    else:
        sl = vol[:, :, index].T
    return sl


def load_display_image(path, bg_tol: int = 25, pad: float = 0.06):
    """Load an image and tightly crop to the object for readable display.

    Raw OmniObject3D renders are 1024x1024 with the object occupying a wide,
    off-center region; a naive center crop shows almost nothing. Here we detect
    the object against the corner background colour and crop to its bounding box
    with a small padding, then return a PIL image.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    if arr.size == 0:
        return img
    bg = arr[0, 0].astype(int)
    mask = np.abs(arr.astype(int) - bg).sum(axis=2) > bg_tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return img
    # Pad, and make the crop square so the object isn't stretched.
    side = int(max(w, h) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    crop = img.crop((left, top, min(img.width, left + side), min(img.height, top + side)))
    return crop


def scatter3d(ax, pts: np.ndarray, color: str, label: str, s: float = 1.2, alpha: float = 0.6):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=color, label=label,
               alpha=alpha, linewidths=0, depthshade=True)


def style_3d_axes(ax, elev: float = 22, azim: float = -60):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.05)
