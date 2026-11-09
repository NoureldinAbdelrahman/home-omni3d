#!/usr/bin/env python3
"""Read-only health check for the local OmniObject3D subset (24-view renders + 4096-pt clouds).

Checks, per category and object:

* categories present vs. the remote (216) and object counts vs. the remote
  (object count = 4096-point HDF5 size / 98304 B; exact on every category we verified);
* renders: exactly 24 PNGs ``000..023.png`` per object, each with a valid PNG header,
  an intact ``IEND`` trailer (catches truncated files) and identical size across views;
  ``transforms.json`` parses and lists the same 24 frames with 4x4 finite matrices;
* point clouds: one ``<obj>.npy`` per render object, shape (4096, 3), float32, finite,
  non-degenerate, and no two objects holding an identical cloud (pairing-bug symptom);
* leftovers: ``.part`` files, ``_downloads/``, ``_extract/``, state-file inconsistencies.

Nothing is modified. ``--remote`` lists remote metadata (no download) to compare counts.
``--deep`` additionally fully decodes every PNG (slow: ~140k images).

Usage::

    python check_dataset.py --dataset-dir dataset --remote
    python check_dataset.py --dataset-dir dataset --remote --deep --jobs 8

Exit code 0 = complete, 1 = problems found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

VIEWS = 24
POINTS = 4096
PNG_SIG = b"\x89PNG\r\n\x1a\n"
IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
BYTES_PER_CLOUD_HDF5 = POINTS * 3 * 8  # float64 xyz per object


class Report:
    def __init__(self):
        self.errors: dict[str, list[str]] = defaultdict(list)
        self.warnings: dict[str, list[str]] = defaultdict(list)

    def error(self, cat: str, msg: str) -> None:
        self.errors[cat].append(msg)

    def warn(self, cat: str, msg: str) -> None:
        self.warnings[cat].append(msg)


def png_info(path: Path) -> tuple[int, int] | str:
    """(width, height) from the IHDR, or an error string. Also checks the IEND trailer."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24 or head[:8] != PNG_SIG or head[12:16] != b"IHDR":
                return "bad PNG header"
            f.seek(max(0, size - 12))
            if f.read(12) != IEND:
                return "truncated (no IEND trailer)"
        return struct.unpack(">II", head[16:24])
    except OSError as exc:
        return f"unreadable: {exc}"


def deep_decode(path: str) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"decode failed: {exc}"


def check_object(obj_dir: Path, deep: bool) -> list[str]:
    issues: list[str] = []
    names = {p.name for p in obj_dir.iterdir()}
    want = {f"{i:03d}.png" for i in range(VIEWS)}
    pngs = {n for n in names if n.endswith(".png")}
    if pngs != want:
        miss, extra = sorted(want - pngs), sorted(pngs - want)
        issues.append(f"{obj_dir.name}: PNG set wrong (missing {len(miss)} {miss[:3]}, extra {extra[:3]})")
    sizes = set()
    for n in sorted(pngs & want):
        info = png_info(obj_dir / n)
        if isinstance(info, str):
            issues.append(f"{obj_dir.name}/{n}: {info}")
        else:
            sizes.add(info)
            if deep and (err := deep_decode(str(obj_dir / n))):
                issues.append(f"{obj_dir.name}/{n}: {err}")
    if len(sizes) > 1:
        issues.append(f"{obj_dir.name}: views differ in resolution {sorted(sizes)}")
    tj = obj_dir / "transforms.json"
    if not tj.is_file():
        issues.append(f"{obj_dir.name}: transforms.json missing")
    else:
        try:
            t = json.loads(tj.read_text())
            frames = t["frames"]
            if [f["file_path"] for f in frames] != [f"{i:03d}.png" for i in range(VIEWS)]:
                issues.append(f"{obj_dir.name}: transforms.json frames != 000..023.png")
            for f in frames:
                m = np.asarray(f["transform_matrix"], dtype=float)
                if m.shape != (4, 4) or not np.isfinite(m).all():
                    issues.append(f"{obj_dir.name}: bad transform_matrix in {f['file_path']}")
                    break
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f"{obj_dir.name}: transforms.json invalid ({exc})")
    return issues


def check_cloud(path: Path) -> tuple[list[str], str | None]:
    try:
        a = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        return [f"{path.stem}: unreadable npy ({exc})"], None
    issues = []
    if a.shape != (POINTS, 3):
        issues.append(f"{path.stem}: shape {a.shape} (expected ({POINTS}, 3))")
    if a.dtype != np.float32:
        issues.append(f"{path.stem}: dtype {a.dtype} (expected float32)")
    if not np.isfinite(a).all():
        issues.append(f"{path.stem}: NaN/inf values")
    elif a.size and float(np.ptp(a, axis=0).max()) < 1e-6:
        issues.append(f"{path.stem}: degenerate cloud (all points identical)")
    return issues, hashlib.sha1(a.tobytes()).hexdigest()


def remote_expectations() -> tuple[dict[str, int], dict[str, int]]:
    """({cat: archive bytes}, {cat: expected objects}) from remote metadata only."""
    import download_bedroom_subset as base
    import download_remaining as dr

    base.ensure_login()
    arch = {
        Path(r.name).name[: -len(".tar.gz")]: r.size
        for r in dr.list_remote(base.RENDER_PREFIX)
        if r.name.endswith(".tar.gz")
    }
    h = {
        Path(r.name).name[: -len("_4096.hdf5")]: r.size
        for r in dr.list_remote("/raw/point_clouds/hdf5_files/4096")
        if r.name.endswith("_4096.hdf5")
    }
    return arch, {c: round(h[c] / BYTES_PER_CLOUD_HDF5) for c in h}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--remote", action="store_true", help="compare with remote category/object counts (lists metadata only)")
    ap.add_argument("--deep", action="store_true", help="fully decode every PNG (slow)")
    ap.add_argument("--jobs", type=int, default=16, help="I/O threads")
    ap.add_argument("--report", type=Path, default=None, help="write a JSON report (default: <dataset-dir>/health_report.json)")
    args = ap.parse_args()

    ds = args.dataset_dir
    rep = Report()
    t0 = time.time()
    renders_root, pc_root = ds / "renders", ds / "point_clouds"
    if not renders_root.is_dir() or not pc_root.is_dir():
        print(f"ERROR: {ds} has no renders/ and point_clouds/")
        return 2

    arch_sizes: dict[str, int] = {}
    expected: dict[str, int] = {}
    if args.remote:
        print("listing remote metadata ...")
        arch_sizes, expected = remote_expectations()
        print(f"remote: {len(arch_sizes)} categories, {sum(expected.values())} objects expected")

    # documented exclusions (manifest.json next to renders/): known upstream gaps, not errors
    excluded: set[tuple[str, str]] = set()
    missing_cats: set[str] = set()
    manifest_p = ds / "manifest.json"
    if manifest_p.is_file():
        mf = json.loads(manifest_p.read_text())
        excluded = {(e["category"], e["object"]) for e in mf.get("excluded_objects", [])}
        missing_cats = {m["category"] for m in mf.get("missing_categories", [])}
        for c in missing_cats:  # documented as absent: do not expect them
            arch_sizes.pop(c, None)
            expected.pop(c, None)
        for c, _ in excluded:
            if c in expected:
                expected[c] -= 1
        print(f"manifest.json: {len(excluded)} excluded object(s), {len(missing_cats)} absent categor(y/ies) documented")

    r_cats = sorted(p.name for p in renders_root.iterdir() if p.is_dir())
    p_cats = sorted(p.name for p in pc_root.iterdir() if p.is_dir())

    # ---- category-level completeness
    if args.remote:
        for c in sorted(set(arch_sizes) - set(r_cats)):
            rep.error(c, "renders: category MISSING locally")
        for c in sorted(set(arch_sizes) - set(p_cats)):
            rep.error(c, "point_clouds: category MISSING locally")
        for c in sorted(set(r_cats) - set(arch_sizes)):
            rep.warn(c, "renders: category not on the remote (extra)")
    for c in sorted(set(r_cats) ^ set(p_cats)):
        where = "renders only" if c in r_cats else "point_clouds only"
        rep.warn(c, f"present in {where}")

    # ---- per-object checks
    n_obj = n_png = n_cloud = 0
    cloud_hash: dict[str, list[str]] = defaultdict(list)
    jobs: list[tuple[str, Path]] = []
    obj_by_cat: dict[str, list[str]] = {}
    for c in r_cats:
        objs = sorted(p for p in (renders_root / c).iterdir() if p.is_dir())
        obj_by_cat[c] = [o.name for o in objs]
        n_obj += len(objs)
        if args.remote and c in expected and len(objs) != expected[c]:
            rep.error(c, f"renders: {len(objs)} objects, remote has {expected[c]}")
        bad_names = [o.name for o in objs if not o.name.startswith(f"{c}_")]
        if bad_names:
            rep.error(c, f"renders: unexpected folder names {bad_names[:3]}")
        jobs += [(c, o) for o in objs]

    print(f"checking {n_obj} render objects ({n_obj * VIEWS} images){' [deep decode]' if args.deep else ''} ...")
    done = 0
    with ThreadPoolExecutor(args.jobs) as ex:
        for (c, _), issues in zip(jobs, ex.map(lambda j: check_object(j[1], args.deep), jobs)):
            n_png += VIEWS
            for msg in issues:
                rep.error(c, "renders: " + msg)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} objects  ({time.time() - t0:.0f}s)")

    print("checking point clouds ...")
    for c in p_cats:
        files = sorted((pc_root / c).glob("*.npy"))
        stems = {f.stem for f in files}
        n_cloud += len(files)
        leftovers = [p.name for p in (pc_root / c).iterdir() if not p.name.endswith(".npy") or p.name.endswith(".tmp")]
        if leftovers:
            rep.error(c, f"point_clouds: stray files {leftovers[:3]}")
        r_objs = set(obj_by_cat.get(c, []))
        exp = expected.get(c)
        if exp is not None and len(files) != exp:
            rep.error(c, f"point_clouds: {len(files)} clouds, remote has {exp}")
        only_r, only_p = sorted(r_objs - stems), sorted(stems - r_objs)
        if c in obj_by_cat and (only_r or only_p):
            # a gap that the remote itself has (clouds == remote count) is upstream, not our download
            upstream = exp is not None and len(files) == exp
            (rep.warn if upstream else rep.error)(
                c, f"pairing: renders without cloud {only_r[:3]}, clouds without renders {only_p[:3]}"
                + (" [matches remote: upstream gap]" if upstream else "")
            )
        with ThreadPoolExecutor(args.jobs) as ex:
            for f, (issues, h) in zip(files, ex.map(check_cloud, files)):
                for msg in issues:
                    rep.error(c, "point_clouds: " + msg)
                if h:
                    cloud_hash[h].append(f"{c}/{f.stem}")
    for names in cloud_hash.values():
        if len(names) > 1:
            rep.error(names[0].split("/")[0], f"point_clouds: identical cloud shared by {names[:4]} (pairing bug?)")

    # ---- pairing: every cloud must be the cloud of the object it is named after.
    # fix_point_cloud_names.json holds a SHA-1 of each object's true cloud (from the named PLY archive).
    map_p = Path(__file__).resolve().parent / "fix_point_cloud_names.json"
    n_paired = 0
    if map_p.is_file():
        from fix_point_cloud_names import cloud_hash

        mapping = json.loads(map_p.read_text())
        print("verifying cloud <-> object pairing against fix_point_cloud_names.json ...")
        by_hash = {h: (c, o) for c, objs in mapping.items() for o, h in objs.items()}
        for c in p_cats:
            if c not in mapping:
                rep.warn(c, "pairing: category not in fix_point_cloud_names.json (cannot verify)")
                continue
            for f in sorted((pc_root / c).glob("*.npy")):
                want = mapping[c].get(f.stem)
                got = cloud_hash(np.load(f))
                if want == got:
                    n_paired += 1
                elif want is None:
                    rep.warn(c, f"pairing: {f.stem} not in the mapping (cannot verify)")
                else:
                    who = by_hash.get(got)
                    rep.error(c, f"pairing: {f.stem}.npy holds "
                              + (f"the cloud of {who[1]}" if who else "an unknown cloud")
                              + " - run fix_point_cloud_names.py")
    else:
        rep.warn("(pairing)", "fix_point_cloud_names.json not found next to check_dataset.py: pairing NOT verified")

    # ---- leftovers + state file
    for scratch in ("_downloads", "_extract"):
        d = ds / scratch
        if d.exists():
            left = [p.name for p in d.iterdir()]
            if left:
                rep.warn("(scratch)", f"{scratch}/ not empty: {left[:4]}")
    state_p = ds / ".download_state.json"
    if state_p.is_file():
        state = json.loads(state_p.read_text())
        for c in state.get("renders", {}):
            if c not in r_cats:
                rep.error(c, "state says renders finished but the folder is missing")
        for c, rec in state.get("renders", {}).items():
            for w in rec.get("warnings", []):
                rep.warn(c, "recorded at extraction: " + w)
    for p in ds.rglob("*.part"):
        rep.warn("(scratch)", f"partial download left: {p.relative_to(ds)}")

    # ---- summary
    size = sum(f.stat().st_size for f in renders_root.rglob("*") if f.is_file())
    print("\n" + "=" * 72)
    print(f"categories: renders {len(r_cats)}, point_clouds {len(p_cats)}"
          + (f" (remote {len(arch_sizes)})" if args.remote else ""))
    print(f"objects: {n_obj}" + (f" (remote {sum(expected.values())})" if args.remote else "")
          + f" | images {n_png} | clouds {n_cloud} | renders on disk {size / 2**30:.1f} GiB"
          + (f" (remote archives {sum(arch_sizes.values()) / 2**30:.1f} GiB)" if args.remote else ""))
    print(f"pairing verified by hash: {n_paired} of {n_cloud} clouds")
    for title, store in (("ERRORS", rep.errors), ("WARNINGS", rep.warnings)):
        total = sum(len(v) for v in store.values())
        print(f"\n{title}: {total} in {len(store)} categories")
        for c in sorted(store):
            for m in store[c][:6]:
                print(f"  [{c}] {m}")
            if len(store[c]) > 6:
                print(f"  [{c}] ... and {len(store[c]) - 6} more")
    ok = not rep.errors
    print("\n" + ("DATASET COMPLETE" if ok else "DATASET INCOMPLETE / PROBLEMS FOUND") + f"  ({time.time() - t0:.0f}s)")

    out = args.report or ds / "health_report.json"
    out.write_text(json.dumps({"complete": ok, "errors": rep.errors, "warnings": rep.warnings,
                               "objects": n_obj, "images": n_png, "clouds": n_cloud}, indent=1))
    print(f"report: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
