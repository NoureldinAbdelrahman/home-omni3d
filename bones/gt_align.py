"""Map OmniObject3D GT point clouds into the render/camera frame of transforms.json.

The GT clouds are in raw scan units; the Blender renders are Z-up and normalised
so the object's largest half-extent is ~0.4. Most scans are Y-up and centred
near their bounding box, but the renders were normalised from the original mesh
(not shipped), so no closed-form rule reproduces the placement exactly. Instead
a similarity (rotation, scale, offset) is fitted per object so the projected
cloud matches the alpha silhouettes of all 24 views under the GT cameras. Only
photos + GT cameras are used - never a reconstruction under evaluation.

    python bones/gt_align.py                      # all objects -> bones/gt_alignment.json
    python bones/gt_align.py --objects chair/chair_001
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0])
RES = 256  # silhouette resolution used for fitting
RELIABLE_INSIDE = 0.9  # below this the fitted alignment is not trusted


def _axis_rotations():
    """The 24 axis-aligned rotations, Y-up -> Z-up first."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3)); M[range(3), perm] = sg
            if np.linalg.det(M) > 0:
                out.append(M)
    out.sort(key=lambda M: not np.allclose(M, YUP_TO_ZUP))
    return out


def load_cameras(obj_dir: Path):
    """Returns names, c2w (N,4,4, OpenCV axes), focal (px at native res), native size."""
    meta = json.loads((obj_dir / "transforms.json").read_text())
    names, c2w = [], []
    for f in meta["frames"]:
        T = np.array(f["transform_matrix"], dtype=np.float64)
        T[:3, :3] = T[:3, :3] @ BLENDER_TO_CV
        names.append(Path(f["file_path"]).stem + ".png"); c2w.append(T)
    size = Image.open(obj_dir / names[0]).size[0]
    return names, np.stack(c2w), 0.5 * size / np.tan(0.5 * meta["camera_angle_x"]), size


def project(P, T, f, size):
    X = (P - T[:3, 3]) @ T[:3, :3]
    return np.stack([f * X[:, 0] / X[:, 2] + size / 2, f * X[:, 1] / X[:, 2] + size / 2], 1)


def _silhouettes(obj_dir, names):
    sil = []
    for n in names:
        a = np.asarray(Image.open(obj_dir / n).getchannel("A").resize((RES, RES), Image.BILINEAR)) > 127
        ys, xs = np.nonzero(a)
        pix = np.stack([xs, ys], 1) + 0.5
        sel = np.random.default_rng(0).choice(len(pix), min(len(pix), 1500), replace=False)
        sil.append((distance_transform_edt(~a), pix[sel]))
    return sil


def apply_alignment(P: np.ndarray, info: dict) -> np.ndarray:
    """Raw GT cloud -> render frame, using a stored fit."""
    R = np.array(info["rotation"])
    return (P - np.array(info["center"])) @ R.T * info["scale"] + np.array(info["offset"])


def fit_gt_to_render(obj_dir: Path, pc_path: Path):
    """Return (P_render (N,3), info) with info usable by :func:`apply_alignment`."""
    names, c2w, f, size = load_cameras(obj_dir)
    P = np.load(pc_path).astype(np.float64)
    c0 = (P.max(0) + P.min(0)) / 2
    P = P - c0
    s0 = 0.4 / np.abs(P).max()
    sil = _silhouettes(obj_dir, names)
    fr = f * RES / size

    def place(x, R0):
        R = Rotation.from_rotvec(x[4:7]).as_matrix() @ R0
        return P @ R.T * s0 * np.exp(x[0]) + x[1:4]

    def cost(x, R0, views=slice(None)):
        Q, c = place(x, R0), 0.0
        for T, (dt, pix) in zip(c2w[views], sil[views]):
            uv = project(Q, T, fr, RES)
            ij = np.clip(uv.astype(int), 0, RES - 1)
            out = dt[ij[:, 1], ij[:, 0]] + np.linalg.norm(uv - np.clip(uv, 0, RES), axis=1)
            c += out.mean() + cKDTree(uv).query(pix)[0].mean()
        return c / len(c2w[views])

    # coarse: every axis rotation x a few scales on 6 views, then refine the best 3
    coarse = []
    for R0 in _axis_rotations():
        for ls in np.log([0.8, 0.95, 1.1]):
            x = np.array([ls, 0, 0, 0, 0, 0, 0.0])
            coarse.append((cost(x, R0, slice(0, 24, 4)), x, R0))
    coarse.sort(key=lambda r: r[0])
    opts = dict(xatol=1e-4, fatol=1e-4, maxiter=1500)
    cands = [(minimize(cost, x, args=(R0,), method="Nelder-Mead", options=opts), R0) for _, x, R0 in coarse[:3]]
    best, R0 = min(cands, key=lambda r: r[0].fun)
    Q = place(best.x, R0)
    inside = []
    for T, (dt, _) in zip(c2w, sil):
        ij = np.clip(project(Q, T, fr, RES).astype(int), 0, RES - 1)
        inside.append((dt[ij[:, 1], ij[:, 0]] <= 1).mean())
    R = Rotation.from_rotvec(best.x[4:7]).as_matrix() @ R0
    return Q, dict(center=c0.tolist(), rotation=R.tolist(), scale=float(s0 * np.exp(best.x[0])),
                   offset=best.x[1:4].tolist(), sil_cost_px=float(best.fun),
                   frac_inside=float(np.mean(inside)))


def _job(obj):
    try:
        _, info = fit_gt_to_render(ROOT / "dataset/renders" / obj, ROOT / "dataset/point_clouds" / f"{obj}.npy")
        return obj, info
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal for the batch
        return obj, {"error": str(exc)}


def all_objects():
    out = []
    for cat in sorted(p.name for p in (ROOT / "dataset/renders").iterdir() if p.is_dir()):
        for obj_dir in sorted((ROOT / "dataset/renders" / cat).iterdir()):
            if (obj_dir / "transforms.json").is_file() and \
                    (ROOT / "dataset/point_clouds" / cat / f"{obj_dir.name}.npy").is_file():
                out.append(f"{cat}/{obj_dir.name}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", help="<category>/<object_id> (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "bones/gt_alignment.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    done = json.loads(a.out.read_text()) if a.out.is_file() else {}
    todo = [o for o in (a.objects or all_objects()) if o not in done or "error" in done[o]]
    print(f"{len(done)} cached, {len(todo)} to fit")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (obj, info) in enumerate(ex.map(_job, todo, chunksize=1), 1):
            done[obj] = info
            if i % 50 == 0 or i == len(todo):
                a.out.write_text(json.dumps(dict(sorted(done.items())), indent=1))
                print(f"{i}/{len(todo)}", flush=True)
    fi = np.array([v["frac_inside"] for v in done.values() if "frac_inside" in v])
    print(f"fitted {len(fi)}; reliable (inside >= {RELIABLE_INSIDE}): {(fi >= RELIABLE_INSIDE).sum()}; "
          f"errors: {sum('error' in v for v in done.values())}")
