#!/usr/bin/env python3
"""Convert the local OmniObject3D bedroom subset into Pix2Vox and AtlasNet layouts."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bedroom_categories import BEDROOM_CATEGORIES  # noqa: E402


def discover_categories(dataset_dir: Path) -> list[str]:
    """Categories present under point_clouds/ and/or renders/ (bedroom first)."""
    found: set[str] = set()
    for sub in ("point_clouds", "renders"):
        root = dataset_dir / sub
        if root.is_dir():
            found |= {p.name for p in root.iterdir() if p.is_dir()}
    ordered = [c for c in BEDROOM_CATEGORIES if c in found]
    ordered += sorted(found - set(ordered))
    return ordered


def list_objects(dataset_dir: Path, category: str) -> list[str]:
    pc_dir = dataset_dir / "point_clouds" / category
    if not pc_dir.is_dir():
        return []
    return sorted(p.stem for p in pc_dir.glob("*.npy"))


def make_splits(object_ids: list[str], seed: int = 0):
    """Deterministic 70/15/15 split; guarantees >=1 train sample."""
    rng = np.random.default_rng(seed)
    ids = list(object_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return [], [], []
    if n == 1:
        return ids, [], []
    if n == 2:
        return [ids[0]], [], [ids[1]]
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - n_train - n_test)
        if n_train + n_val + n_test > n:
            n_test = max(0, n - n_train - n_val)
    test = ids[:n_test]
    val = ids[n_test:n_test + n_val]
    train = ids[n_test + n_val:]
    if not train:
        train = [ids[-1]]
        if test:
            test = test[:-1]
        elif val:
            val = val[:-1]
    return sorted(train), sorted(val), sorted(test)


def points_to_voxel(points: np.ndarray, grid: int = 32) -> np.ndarray:
    pts = points.astype(np.float64)
    mins = pts.min(0)
    maxs = pts.max(0)
    spans = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    pts = (pts - mins) / spans * (grid - 1)
    idx = np.clip(pts.astype(np.int64), 0, grid - 1)
    vox = np.zeros((grid, grid, grid), dtype=np.float32)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # Densify: single 6-connectivity dilation approximates surface occupancy.
    try:
        from scipy import ndimage
        vox = ndimage.binary_dilation(vox, iterations=1).astype(np.float32)
    except ImportError:
        pass
    return vox


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------- Pix2Vox ----------------

def prepare_pix2vox(
    dataset_dir: Path,
    out_dir: Path,
    grid: int = 32,
    categories: list[str] | None = None,
) -> None:
    rendering_root = out_dir / "OmniObjectRendering"
    voxel_root = out_dir / "OmniObjectVox32"
    taxonomy_path = out_dir / "OmniObject3D.json"

    cats = categories if categories is not None else discover_categories(dataset_dir)
    taxonomy = []
    n_obj = n_img = n_vox = 0
    for cat in cats:
        object_ids = list_objects(dataset_dir, cat)
        if not object_ids:
            continue
        train, val, test = make_splits(object_ids)
        taxonomy.append({
            "taxonomy_id": cat,
            "taxonomy_name": cat,
            "train": train,
            "val": val,
            "test": test,
        })
        for obj in object_ids:
            pc_path = dataset_dir / "point_clouds" / cat / f"{obj}.npy"
            points = np.load(pc_path)
            vox = points_to_voxel(points, grid=grid)
            mat_path = voxel_root / cat / obj / "model.mat"
            mat_path.parent.mkdir(parents=True, exist_ok=True)
            from scipy.io import savemat
            savemat(str(mat_path), {"Volume": vox})
            n_vox += 1

            render_dir = rendering_root / cat / obj / "rendering"
            for src in sorted((dataset_dir / "renders" / cat / obj).glob("*.png")):
                idx = int(src.stem)
                link_or_copy(src, render_dir / f"{idx:02d}.png")
                n_img += 1
            n_obj += 1

    taxonomy_path.write_text(json.dumps(taxonomy, indent=2))
    print(f"[Pix2Vox] objects={n_obj} images={n_img} voxels={n_vox} "
          f"-> {out_dir}")


# ---------------- AtlasNet ----------------

def prepare_atlasnet(
    dataset_dir: Path,
    out_dir: Path,
    categories: list[str] | None = None,
) -> None:
    """out_dir should be AtlasNet/dataset/data."""
    pc_root = out_dir / "ShapeNetV1PointCloud"
    render_root = out_dir / "ShapeNetV1Renderings"
    taxonomy_path = out_dir / "taxonomy.json"

    cats = categories if categories is not None else discover_categories(dataset_dir)
    taxonomy = []
    n_obj = n_img = n_pc = 0
    for cat in cats:
        object_ids = list_objects(dataset_dir, cat)
        if not object_ids:
            continue
        taxonomy.append({
            "synsetId": cat,
            # AtlasNet takes name.split(',')[0] as the friendly class name.
            "name": f"{cat}, house {cat}",
        })
        for obj in object_ids:
            src_pc = dataset_dir / "point_clouds" / cat / f"{obj}.npy"
            dst_pc = pc_root / cat / f"{obj}.npy"
            link_or_copy(src_pc, dst_pc)
            n_pc += 1

            for src in sorted((dataset_dir / "renders" / cat / obj).glob("*.png")):
                idx = int(src.stem)
                link_or_copy(src, render_root / cat / obj / "rendering" / f"{idx:02d}.png")
                n_img += 1
            n_obj += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2))
    print(f"[AtlasNet] objects={n_obj} images={n_img} pointclouds={n_pc} "
          f"-> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--pix2vox-out", type=Path,
                    default=ROOT / "Pix2Vox" / "datasets" / "OmniObject3D")
    ap.add_argument("--atlasnet-out", type=Path,
                    default=ROOT / "AtlasNet" / "dataset" / "data")
    ap.add_argument("--target", choices=["pix2vox", "atlasnet", "both"],
                    default="both")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--categories", nargs="+", metavar="CAT", default=None,
                    help="Override category list (default: all found on disk).")
    args = ap.parse_args()

    cats = None
    if args.categories:
        from bedroom_categories import resolve_category
        cats = [resolve_category(c) for c in args.categories]
    elif args.dataset_dir.is_dir():
        found = discover_categories(args.dataset_dir)
        if found:
            print(f"Discovered {len(found)} categories under {args.dataset_dir}")
            cats = found

    if args.target in ("pix2vox", "both"):
        prepare_pix2vox(args.dataset_dir, args.pix2vox_out, grid=args.grid,
                        categories=cats)
    if args.target in ("atlasnet", "both"):
        prepare_atlasnet(args.dataset_dir, args.atlasnet_out, categories=cats)


if __name__ == "__main__":
    main()
