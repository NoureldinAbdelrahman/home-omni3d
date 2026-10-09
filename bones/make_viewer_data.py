#!/usr/bin/env python3
"""Pack COLMAP results into compact files for the 3D web viewer (bones/viewer/).

Writes to ``--out`` (default bones/viewer/data):
* manifest.json        - every object's metrics (known + sfm), cameras, offsets
* pts/<cat>.json       - {"b64": base64 of int16 xyz} (scale 1/32767, render frame) for GT, known, sfm
* img/<cat>.webp       - one 256px photo per object, in a horizontal strip
"""
import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_align import RELIABLE_INSIDE, ROOT, apply_alignment, load_cameras  # noqa: E402

THUMB = 256
KEYS = ("registered", "n_pred", "chamfer_l1", "precision_1", "recall_1", "f_1", "precision_2", "recall_2",
        "f_2", "precision_5", "recall_5", "f_5", "f_ceiling_1", "f_ceiling_2", "f_ceiling_5")


def read_metrics(mode):
    rows = {}
    for r in csv.DictReader(open(ROOT / f"results/colmap/sparse_{mode}/metrics.csv")):
        rows[r["object"]] = {"status": r["status"]} | {
            k: (round(float(r[k]), 4) if r.get(k) not in (None, "") else None) for k in KEYS}
    return rows


def sfm_registered(ws: Path, names):
    """Which of the 24 views the largest blind-SfM model registered."""
    best = None
    for d in sorted((ws / "sparse").glob("*")):
        if (d / "images.bin").is_file():
            rec = pycolmap.Reconstruction(str(d))
            if best is None or rec.num_reg_images() > best.num_reg_images():
                best = rec
    reg = {im.name for im in best.images.values() if im.has_pose} if best else set()
    return [n in reg for n in names]


def quant(P):
    return np.clip(np.round(np.asarray(P, np.float64) * 32767), -32767, 32767).astype("<i2").reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "bones/viewer/data")
    a = ap.parse_args()
    (a.out / "pts").mkdir(parents=True, exist_ok=True)
    (a.out / "img").mkdir(parents=True, exist_ok=True)

    known, sfm = read_metrics("known"), read_metrics("sfm")
    align = json.loads((ROOT / "bones/gt_alignment.json").read_text())
    summaries = {m: json.loads((ROOT / f"results/colmap/sparse_{m}/summary.json").read_text())
                 for m in ("known", "sfm")}

    by_cat = {}
    for o in sorted(known):
        by_cat.setdefault(o.split("/")[0], []).append(o)

    objects, cats = [], []
    for cat, objs in sorted(by_cat.items()):
        chunks, offset, strip = [], 0, Image.new("RGBA", (THUMB * len(objs), THUMB))
        for i, o in enumerate(objs):
            obj_dir = ROOT / "dataset/renders" / o
            names, c2w, _, _ = load_cameras(obj_dir)
            ent = dict(id=o, cat=cat, thumb=i, cams=np.round(c2w[:, :3, 3], 4).tolist(),
                       known=known[o], sfm=sfm.get(o), offsets={},
                       gt_fit=round(align.get(o, {}).get("frac_inside", 0), 3))
            ent["gt_reliable"] = ent["gt_fit"] >= RELIABLE_INSIDE
            if o in align and "rotation" in align[o]:
                gt = apply_alignment(np.load(ROOT / "dataset/point_clouds" / f"{o}.npy").astype(np.float64), align[o])
            else:
                gt = np.zeros((0, 3))
            layers = {"gt": gt}
            for mode in ("known", "sfm"):
                p = ROOT / "bones/work" / mode / o.replace("/", "__") / "points.npy"
                layers[mode] = np.load(p) if p.is_file() else np.zeros((0, 3))
            for k, P in layers.items():
                q = quant(P)
                ent["offsets"][k] = [offset, len(q)]
                chunks.append(q); offset += len(q)
            ent["sfm_reg"] = sfm_registered(ROOT / "bones/work/sfm" / o.replace("/", "__"), names)
            im = Image.open(obj_dir / names[0]).convert("RGBA")
            im.thumbnail((THUMB, THUMB))
            strip.paste(im, (THUMB * i, 0))
            objects.append(ent)
        raw = np.concatenate(chunks).tobytes()  # artifacts serve no raw binary, so base64 in JSON
        (a.out / "pts" / f"{cat}.json").write_text(json.dumps({"b64": base64.b64encode(raw).decode()}))
        strip.save(a.out / "img" / f"{cat}.webp", quality=82, method=6)
        ok = [e for e in objects if e["cat"] == cat]
        cats.append(dict(name=cat, n=len(objs),
                         f5_known=summaries["known"]["per_category"].get(cat, {}).get("f_5"),
                         f5_sfm=summaries["sfm"]["per_category"].get(cat, {}).get("f_5"),
                         reg_sfm=float(np.mean([e["sfm"]["registered"] or 0 for e in ok if e["sfm"]]))))
        print(cat, len(objs), offset, "points", flush=True)

    manifest = dict(
        scale=1 / 32767, thumb=THUMB,
        summary={m: {k: summaries[m][k] for k in summaries[m] if k != "per_category"} for m in summaries},
        categories=cats, objects=objects)
    (a.out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


if __name__ == "__main__":
    main()
