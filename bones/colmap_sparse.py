#!/usr/bin/env python3
"""Sparse COLMAP (pycolmap, CPU) reconstruction of OmniObject3D objects.

Per object: composite RGBA onto gray + alpha mask -> masked SIFT -> exhaustive
matching, then one of

* ``known``: triangulate with the GT cameras from transforms.json (all 24 views
  posed; isolates matching/triangulation quality - the multi-view best case).
* ``sfm``:   blind incremental SfM. Registered cameras are Sim(3)-aligned to the
  GT cameras (scored as pose error) and the same Sim(3) moves the points into
  the render frame.

Points are written in the render frame (same frame as transforms.json) to
``<work>/<mode>/<cat>__<obj>/points.npy`` (+ ``sparse.ply``); per-object stats
go to ``results/colmap/sparse_<mode>/objects.json`` (score with eval_clouds.py).

    python bones/colmap_sparse.py --mode known --per-category 4
    python bones/colmap_sparse.py --mode sfm --objects chair/chair_001
"""
import argparse
import os
import json
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_align import ROOT, all_objects, load_cameras  # noqa: E402


def prepare(obj_dir: Path, ws: Path, names):
    img_dir, mask_dir = ws / "images", ws / "masks"
    img_dir.mkdir(parents=True, exist_ok=True); mask_dir.mkdir(exist_ok=True)
    for n in names:
        rgba = np.asarray(Image.open(obj_dir / n).convert("RGBA"), dtype=np.float32)
        a = rgba[..., 3:] / 255.0
        rgb = rgba[..., :3] * a + 127.0 * (1 - a)
        Image.fromarray(rgb.astype(np.uint8)).save(img_dir / n)
        Image.fromarray(((a[..., 0] > 0.5) * 255).astype(np.uint8)).save(mask_dir / f"{n}.png")
    return img_dir, mask_dir


def umeyama(src, dst):
    """Sim(3) s,R,t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def known_pose_reconstruction(db_path: Path, names, c2w):
    """Reconstruction holding the DB's cameras/images, posed with the GT cameras."""
    db = pycolmap.Database.open(str(db_path))
    rec = pycolmap.Reconstruction()
    for cam in db.read_all_cameras():
        rec.add_camera_with_trivial_rig(cam)
    for im in db.read_all_images():
        new = pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id)
        new.points2D = [pycolmap.Point2D(xy) for xy in db.read_keypoints(im.image_id)[:, :2].astype(float)]
        rec.add_image_with_trivial_frame(new, pycolmap.Rigid3d(np.linalg.inv(c2w[names.index(im.name)])[:3]))
    db.close()
    return rec


def sfm_to_render_frame(rec, names, c2w):
    """Sim(3) from registered SfM cameras to GT cameras + pose errors."""
    idx, cen, rot = [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        T = im.cam_from_world().inverse().matrix()
        idx.append(names.index(im.name)); cen.append(T[:, 3]); rot.append(T[:, :3])
    if len(idx) < 3:
        return None, None
    cen, rot = np.array(cen), np.array(rot)
    gc, gr = c2w[idx, :3, 3], c2w[idx, :3, :3]
    s, R, t = umeyama(cen, gc)
    cerr = np.linalg.norm(s * cen @ R.T + t - gc, axis=1)
    rerr = [np.degrees(np.arccos(np.clip((np.trace(ra.T @ rg) - 1) / 2, -1, 1))) for ra, rg in zip(R @ rot, gr)]
    radius = np.linalg.norm(gc, axis=1).mean()
    pose = dict(center_err_rel=float(np.mean(cerr) / radius), rot_err_deg_mean=float(np.mean(rerr)),
                rot_err_deg_max=float(np.max(rerr)))
    return (s, R, t), pose


def run_object(obj: str, mode: str, work: Path, threads: int = -1):
    obj_dir = ROOT / "dataset/renders" / obj
    ws = work / mode / obj.replace("/", "__")
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    names, c2w, f, size = load_cameras(obj_dir)
    t, t0 = {}, time.perf_counter()

    img_dir, mask_dir = prepare(obj_dir, ws, names)
    db = ws / "database.db"
    ro = pycolmap.ImageReaderOptions()
    ro.mask_path = str(mask_dir)
    ro.camera_model = "PINHOLE"
    ro.camera_params = f"{f},{f},{size / 2},{size / 2}"
    eo = pycolmap.FeatureExtractionOptions()
    eo.num_threads = threads
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = threads
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              extraction_options=eo, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, matching_options=mo, device=pycolmap.Device.cpu)
    t["features_s"] = time.perf_counter() - t0; t0 = time.perf_counter()

    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = threads
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    out = dict(object=obj, mode=mode, n_images=len(names))
    (ws / "sparse").mkdir()
    if mode == "known":
        rec = pycolmap.triangulate_points(known_pose_reconstruction(db, names, c2w), db, img_dir,
                                          ws / "sparse", options=opts)
        to_render = None
    else:
        recs = pycolmap.incremental_mapping(db, img_dir, ws / "sparse", opts)
        out["n_models"] = len(recs)
        rec = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
        to_render, pose = sfm_to_render_frame(rec, names, c2w) if rec else (None, None)
        out["pose_eval"] = pose
    t["reconstruct_s"] = time.perf_counter() - t0

    P = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3) if rec else np.zeros((0, 3))
    if mode == "sfm":
        P = P @ (to_render[0] * to_render[1]).T + to_render[2] if to_render else np.zeros((0, 3))
    np.save(ws / "points.npy", P.astype(np.float32))
    if rec is not None and len(P):
        rec.export_PLY(str(ws / "sparse.ply"))  # SfM frame for sfm mode; render frame for known
    out.update(registered=rec.num_reg_images() if rec else 0, n_points=len(P),
               mean_reproj_err=float(rec.compute_mean_reprojection_error()) if rec and len(P) else None,
               time_s=t)
    for sub in ("images", "masks"):  # keep workspaces small; database + sparse model stay
        shutil.rmtree(ws / sub)
    return out


def _job(args):
    obj, mode, work, threads = args
    try:
        return run_object(obj, mode, work, threads)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal for the batch
        return dict(object=obj, mode=mode, error=str(exc))


def sample_objects(per_category: int, seed: int = 0):
    by_cat = {}
    for o in all_objects():
        by_cat.setdefault(o.split("/")[0], []).append(o)
    rng = random.Random(seed)
    return [o for cat in sorted(by_cat) for o in sorted(rng.sample(by_cat[cat], min(per_category, len(by_cat[cat]))))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "sfm"], default="known")
    ap.add_argument("--objects", nargs="+", help="<category>/<object_id>")
    ap.add_argument("--per-category", type=int, default=4, help="random sample size when --objects is omitted")
    ap.add_argument("--work", type=Path, default=ROOT / "bones/work")
    ap.add_argument("--workers", type=int, default=4, help="parallel objects; cores are split between them")
    a = ap.parse_args()

    objs = a.objects or sample_objects(a.per_category)
    out_path = ROOT / f"results/colmap/sparse_{a.mode}/objects.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["object"]: r for r in json.loads(out_path.read_text())} if out_path.is_file() else {}
    todo = [o for o in objs if o not in done or "error" in done[o]]
    print(f"{len(objs)} objects, {len(todo)} to run ({a.mode})", flush=True)
    threads = max(1, (os.cpu_count() or 1) // a.workers)  # COLMAP otherwise spawns a thread per core per worker
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_job, [(o, a.mode, a.work, threads) for o in todo]), 1):
            done[r["object"]] = r
            print(f"[{i}/{len(todo)}] {r['object']}: " +
                  (r["error"] if "error" in r else f"reg {r['registered']}/24, {r['n_points']} pts"), flush=True)
            out_path.write_text(json.dumps(sorted(done.values(), key=lambda r: r["object"]), indent=1))
