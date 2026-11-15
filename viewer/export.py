"""Export reconstruction results to the team 3D viewer (``python viewer/view.py``).

Every run is shown in one shared frame: the object's ground-truth cloud
(``dataset/point_clouds/<cat>/<obj>.npy``) centred on its bounding-box centre and
scaled so the farthest GT point is at radius 1 (the unit sphere of
docs/METRICS.md). Metrics are computed here, the same way for every run.

    import sys; sys.path.insert(0, "viewer")        # from the repo root
    from export import ViewerRun, normalize_like_gt, voxels_to_points

    run = ViewerRun(owner="nesegemaa", model="AtlasNet", name="svr25",
                    label="AtlasNet SVR, 25 patches", notes="test split, view 0")
    for obj_id, pts in predictions.items():          # obj_id like "chair/chair_001"
        run.add(obj_id, pts)                          # (N,3) already in the GT unit-sphere frame
        # run.add(obj_id, pts, frame="raw")           # or in the raw dataset/point_clouds frame
    run.save()                                        # -> viewer/runs/nesegemaa__atlasnet__svr25/

Voxel models: ``voxels_to_points(vox, obj_id)`` inverts prepare_model_data.points_to_voxel.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
TAUS = (0.01, 0.02, 0.05)
QUANT = 1.5 / 32767  # stored as int16 in [-1.5, 1.5]; the unit sphere fits with margin


def load_gt_raw(obj_id: str) -> np.ndarray:
    return np.load(DATASET / "point_clouds" / f"{obj_id}.npy").astype(np.float64)


def gt_frame(gt_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre, radius) of the shared frame: bbox centre, farthest GT point = 1."""
    c = (gt_raw.max(0) + gt_raw.min(0)) / 2
    return c, float(np.linalg.norm(gt_raw - c, axis=1).max())


def normalize_like_gt(points_raw: np.ndarray, obj_id: str) -> np.ndarray:
    """Raw dataset/point_clouds frame -> shared unit-sphere frame of ``obj_id``."""
    c, r = gt_frame(load_gt_raw(obj_id))
    return (np.asarray(points_raw, np.float64) - c) / r


def voxels_to_points(vox: np.ndarray, obj_id: str, threshold: float = 0.5) -> np.ndarray:
    """Occupied voxel centres -> shared frame. Inverse of prepare_model_data.points_to_voxel,
    whose grid spans the GT bounding box per axis."""
    gt = load_gt_raw(obj_id)
    grid = vox.shape[0]
    mins, maxs = gt.min(0), gt.max(0)
    spans = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    idx = np.argwhere(np.asarray(vox) >= threshold).astype(np.float64)
    raw = mins + (idx + 0.5) / (grid - 1) * spans
    return normalize_like_gt(raw, obj_id)


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Both clouds in the shared frame. Chamfer-L1 and P/R/F at TAUS, plus the
    F ceiling a perfect surface reaches against the 4096-point GT (leave-one-out)."""
    loo = cKDTree(gt).query(gt, k=2)[0][:, 1]
    out = {f"f_ceiling_{int(t * 100)}": float((loo < t).mean()) for t in TAUS}
    out["n_points"] = int(len(pred))
    if not len(pred):
        return out | {f"{k}_{int(t * 100)}": 0.0 for t in TAUS for k in ("precision", "recall", "f")}
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    out["chamfer_l1"] = float(d_pg.mean() + d_gp.mean())
    for t in TAUS:
        p, r = float((d_pg < t).mean()), float((d_gp < t).mean())
        k = int(t * 100)
        out |= {f"precision_{k}": p, f"recall_{k}": r, f"f_{k}": 2 * p * r / (p + r) if p + r else 0.0}
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


class ViewerRun:
    """One model run: predictions for any subset of objects, saved under viewer/runs/."""

    METRICS = ["chamfer_l1", "n_points"] + [f"{k}_{int(t * 100)}" for t in TAUS
                                            for k in ("precision", "recall", "f", "f_ceiling")]

    def __init__(self, owner: str, model: str, name: str, label: str | None = None, notes: str = ""):
        self.id = f"{_slug(owner)}__{_slug(model)}__{_slug(name)}"
        self.meta = dict(id=self.id, owner=owner, model=model, name=name,
                         label=label or f"{model} {name}", notes=notes)
        self.objects: dict[str, dict] = {}
        self._points: dict[str, np.ndarray] = {}

    def add(self, obj_id: str, points: np.ndarray, frame: str = "normalized", info: dict | None = None,
            cameras: np.ndarray | None = None, cameras_used: list[bool] | None = None,
            excluded: str | None = None):
        """Add one object's prediction.

        frame        "normalized" (shared unit-sphere frame) or "raw" (dataset/point_clouds frame).
        info         extra per-object numbers to show (e.g. {"views": 24}); shown as-is.
        cameras      optional (V,3) camera centres in the same frame as ``points``.
        cameras_used optional per-camera flags (drawn filled when True).
        excluded     reason to show the object but leave it out of the scores.
        """
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        cams = None if cameras is None else np.asarray(cameras, np.float64).reshape(-1, 3)
        if frame == "raw":
            c, r = gt_frame(load_gt_raw(obj_id))
            pts = (pts - c) / r
            cams = None if cams is None else (cams - c) / r
        elif frame != "normalized":
            raise ValueError("frame must be 'normalized' or 'raw'")
        ent = dict(info=info or {})
        if cams is not None:
            ent["cameras"] = np.round(cams, 4).tolist()
            if cameras_used is not None:
                ent["cameras_used"] = [bool(x) for x in cameras_used]
        if excluded:
            ent["excluded"] = excluded
        self.objects[obj_id] = ent
        self._points[obj_id] = pts

    def save(self) -> Path:
        out = RUNS_DIR / self.id
        (out / "pts").mkdir(parents=True, exist_ok=True)
        for old in (out / "pts").glob("*.json"):
            old.unlink()
        by_cat: dict[str, list[str]] = {}
        for obj_id in sorted(self.objects):
            by_cat.setdefault(obj_id.split("/")[0], []).append(obj_id)
        for cat, ids in by_cat.items():
            chunks, offset = [], 0
            for obj_id in ids:
                ent, pts = self.objects[obj_id], self._points[obj_id]
                if "excluded" not in ent:
                    c, r = gt_frame(load_gt_raw(obj_id))
                    ent["metrics"] = score(pts, (load_gt_raw(obj_id) - c) / r)
                q = np.clip(np.round(pts / QUANT), -32767, 32767).astype("<i2")
                ent["offset"] = [offset, len(q)]
                chunks.append(q); offset += len(q)
            raw = np.concatenate(chunks).tobytes() if chunks else b""
            (out / "pts" / f"{cat}.json").write_text(json.dumps({"b64": base64.b64encode(raw).decode()}))

        scored = [e["metrics"] for e in self.objects.values() if "metrics" in e]
        per_cat = {}
        for cat, ids in by_cat.items():
            rows = [self.objects[i]["metrics"] for i in ids if "metrics" in self.objects[i]]
            if rows:
                per_cat[cat] = {k: _mean(rows, k) for k in self.METRICS}
        summary = dict(n_objects=len(self.objects), n_scored=len(scored),
                       micro={k: _mean(scored, k) for k in self.METRICS},
                       macro={k: (float(np.mean([v[k] for v in per_cat.values() if v[k] is not None]))
                                  if per_cat else None) for k in self.METRICS})
        doc = self.meta | dict(quant=QUANT, summary=summary, categories=per_cat, objects=self.objects)
        (out / "run.json").write_text(json.dumps(doc, separators=(",", ":")))
        print(f"saved {self.id}: {len(self.objects)} objects, {len(scored)} scored, "
              f"F@5% {summary['micro']['f_5'] if scored else float('nan'):.3f} -> {out.relative_to(ROOT)}")
        return out
