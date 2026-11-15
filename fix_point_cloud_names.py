#!/usr/bin/env python3
"""Repair point-cloud <-> object pairing in an already-downloaded ``dataset/``.

Bug: OmniObject3D's per-category ``<cat>_<N>.hdf5`` stores clouds as unnamed
rows in the authors' filesystem order, not sorted by object ID. The old
downloader named row ``i`` after the ``i``-th *sorted* render folder, so ~95%
of ``point_clouds/<cat>/<obj>.npy`` files hold a different object of the same
category. The cloud data itself is intact - only the file names are shuffled.

This script renames the files back, offline. ``fix_point_cloud_names.json``
lists, per object, a SHA-1 of its true 4096-point cloud (derived from the named
``raw/point_clouds/ply_files/4096_ply.tar.gz``). Every local ``.npy`` is hashed
and renamed to the object whose hash it matches.

Safe to run more than once (an already-fixed dataset is left unchanged), and
a category is skipped untouched if any of its files do not match the mapping.

    python fix_point_cloud_names.py --dry-run   # report only
    python fix_point_cloud_names.py             # apply
    python prepare_model_data.py --target both  # then rebuild model inputs

Maintainers can regenerate the mapping from the named archive with
``--build-mapping path/to/4096_ply.tar.gz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MAPPING_PATH = ROOT / "fix_point_cloud_names.json"


def cloud_hash(points: np.ndarray) -> str:
    """Hash of the cloud as stored by the downloader: (N, 3) little-endian float32."""
    return hashlib.sha1(np.ascontiguousarray(points, dtype="<f4").tobytes()).hexdigest()


def read_ascii_ply(data: bytes) -> np.ndarray:
    header, _, body = data.partition(b"end_header\n")
    if b"format ascii" not in header:
        raise ValueError("expected an ASCII PLY")
    return np.loadtxt(body.decode().splitlines(), dtype=np.float64, usecols=(0, 1, 2), ndmin=2)


def build_mapping(ply_tar: Path, categories: set[str] | None) -> dict[str, dict[str, str]]:
    """``{category: {object_id: hash}}`` from ``<N>/<cat>/<obj>/pcd_<N>.ply`` members."""
    mapping: dict[str, dict[str, str]] = {}
    with tarfile.open(ply_tar, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if not member.isfile() or len(parts) != 4 or not parts[3].endswith(".ply"):
                continue
            cat, obj = parts[1], parts[2]
            if categories is not None and cat not in categories:
                continue
            points = read_ascii_ply(tar.extractfile(member).read())
            mapping.setdefault(cat, {})[obj] = cloud_hash(points)
    return {c: dict(sorted(objs.items())) for c, objs in sorted(mapping.items())}


def plan_category(pc_dir: Path, expected: dict[str, str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Return ``(renames [(src_stem, dst_stem)], errors)`` for one category."""
    by_hash = {h: obj for obj, h in expected.items()}
    if len(by_hash) != len(expected):
        return [], ["mapping has duplicate hashes"]

    local = {p.stem: cloud_hash(np.load(p)) for p in sorted(pc_dir.glob("*.npy"))}
    errors = [f"{stem}.npy matches no known object (modified or foreign file)"
              for stem, h in local.items() if h not in by_hash]
    targets = [by_hash[h] for h in local.values() if h in by_hash]
    if len(set(targets)) != len(targets):
        errors.append("two local files hold the same cloud")
    if errors:
        return [], errors

    renames = [(stem, by_hash[h]) for stem, h in local.items() if by_hash[h] != stem]
    occupied = set(local) - {src for src, _ in renames}
    clashes = sorted({dst for _, dst in renames} & occupied)
    if clashes:
        errors.append(f"target name(s) already hold the right cloud: {clashes}")
    return renames, errors


def apply_renames(pc_dir: Path, renames: list[tuple[str, str]]) -> None:
    """Two-phase move via a temp dir so no file is overwritten mid-way."""
    tmp = pc_dir / ".fix_point_cloud_names_tmp"
    tmp.mkdir(exist_ok=True)
    for src, _ in renames:
        (pc_dir / f"{src}.npy").rename(tmp / f"{src}.npy")
    for src, dst in renames:
        (tmp / f"{src}.npy").rename(pc_dir / f"{dst}.npy")
    tmp.rmdir()


def run(dataset_dir: Path, dry_run: bool) -> int:
    pc_root = dataset_dir / "point_clouds"
    if not pc_root.is_dir():
        print(f"No point_clouds/ under {dataset_dir} - pass --dataset-dir.", file=sys.stderr)
        return 2
    leftovers = sorted(pc_root.glob("*/.fix_point_cloud_names_tmp"))
    if leftovers:
        print("Found leftovers of an interrupted run; move their files back first:", file=sys.stderr)
        for d in leftovers:
            print(f"  {d}", file=sys.stderr)
        return 2
    mapping = json.loads(MAPPING_PATH.read_text())

    n_fixed = n_ok = 0
    failed: list[str] = []
    unknown = sorted(p.name for p in pc_root.iterdir() if p.is_dir() and p.name not in mapping)
    for cat in sorted(p.name for p in pc_root.iterdir() if p.is_dir() and p.name in mapping):
        pc_dir = pc_root / cat
        renames, errors = plan_category(pc_dir, mapping[cat])
        if errors:
            failed.append(cat)
            print(f"[SKIP] {cat}: " + "; ".join(errors))
            continue
        n_total = len(list(pc_dir.glob("*.npy")))
        n_ok += n_total - len(renames)
        n_fixed += len(renames)
        if not renames:
            print(f"[ok]   {cat}: {n_total} file(s) already correct")
            continue
        print(f"[fix]  {cat}: renaming {len(renames)}/{n_total} file(s)")
        if not dry_run:
            apply_renames(pc_dir, renames)
            check, check_errors = plan_category(pc_dir, mapping[cat])
            if check or check_errors:
                print(f"[FAIL] {cat}: verification failed after rename", file=sys.stderr)
                return 1

    verb = "would rename" if dry_run else "renamed"
    print(f"\n{verb} {n_fixed} file(s); {n_ok} already correct; {len(failed)} category(ies) skipped.")
    if unknown:
        print(f"Not in mapping (left untouched): {', '.join(unknown)}")
    if failed:
        print("Skipped categories were left untouched - check those files against the shared copy.")
    if n_fixed and not dry_run:
        print("Now rebuild model inputs:  python prepare_model_data.py --target both")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change; touch nothing.")
    ap.add_argument("--build-mapping", type=Path, metavar="PLY_TAR",
                    help="Regenerate the mapping JSON from <N>_ply.tar.gz (maintainers only).")
    args = ap.parse_args()

    if args.build_mapping:
        pc_root = args.dataset_dir / "point_clouds"
        cats = {p.name for p in pc_root.iterdir() if p.is_dir()} if pc_root.is_dir() else None
        mapping = build_mapping(args.build_mapping, cats)
        MAPPING_PATH.write_text(json.dumps(mapping, indent=1) + "\n")
        print(f"Wrote {MAPPING_PATH.name}: {len(mapping)} categories, "
              f"{sum(map(len, mapping.values()))} objects")
        return 0
    return run(args.dataset_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
