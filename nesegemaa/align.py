"""Shared point-cloud helpers for the nesegemaa track (CPU-only, scipy+numpy).

- kabsch / icp_align: rigid alignment of a prediction onto GT. Used for
  *visualization* variants (see export_to_viewer --align) and pose-gap
  analysis. Aligned numbers are NOT a substitute for raw metrics: alignment
  hides pose error, so aligned runs are always exported under a separate
  viewer run id with the transform recorded per object.
- subsample: deterministic cap on points shown per object (seeded by object
  id), so 64k-point AtlasNet-SPHERE clouds don't out-draw 4k GT clouds.
"""

import hashlib

import numpy as np
from scipy.spatial import cKDTree


def subsample(points: np.ndarray, n: int, key: str) -> np.ndarray:
    """Deterministic random subset of rows (no-op when len <= n or n <= 0)."""
    if n <= 0 or len(points) <= n:
        return points
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False)]


def kabsch(src: np.ndarray, dst: np.ndarray):
    """Optimal rigid (R, t) mapping src -> dst (paired rows)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R, cd - R @ cs


def icp_align(src: np.ndarray, dst: np.ndarray, iters: int = 25):
    """Point-to-point ICP of src onto dst. Returns (aligned, T_4x4).

    Local optimizer: the returned gap is a *lower bound* on the rigid pose
    error (it can get stuck, never invent alignment that isn't there).
    """
    cur = np.asarray(src, dtype=np.float64).copy()
    T = np.eye(4)
    for _ in range(iters):
        _, idx = cKDTree(dst).query(cur)
        R, t = kabsch(cur, dst[idx])
        cur = cur @ R.T + t
        Ti = np.eye(4)
        Ti[:3, :3], Ti[:3, 3] = R, t
        T = Ti @ T
    return cur, T
