#!/usr/bin/env python3
"""Download extra house-related OmniObject3D categories (not the original 18).

On a fresh lab PC this pulls kitchen / living-room / bathroom / utility objects
so you can start Pix2Vox / AtlasNet training immediately. Copy the bedroom
``dataset/`` tree later and re-run ``prepare_model_data.py`` to merge.

Usage::

    # plan only
    python download_house_extra.py --dry-run

    # download ~32 new categories into dataset/ (skips anything already local)
    python download_house_extra.py --output-dir dataset --point-count 4096

    # force re-download a specific list
    python download_house_extra.py --categories kettle speaker soap --force

    # use published OpenDataLab keys if you have no AK/SK
    python download_house_extra.py --use-fallback-keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from bedroom_categories import resolve_category
from house_extra_categories import EXTRA_HOUSE_CATEGORIES
from download_bedroom_subset import (
    build_arg_parser as _base_parser,
    run as _run_base,
)


def local_categories(output_dir: Path) -> set[str]:
    """Categories already present under point_clouds/ and/or renders/."""
    found: set[str] = set()
    for sub in ("point_clouds", "renders"):
        root = output_dir / sub
        if root.is_dir():
            found |= {p.name for p in root.iterdir() if p.is_dir()}
    return found


def pick_categories(
    requested: Sequence[str] | None,
    output_dir: Path,
    *,
    force: bool,
    include_bedroom: bool,
) -> list[str]:
    if requested:
        cats = [resolve_category(c) for c in requested]
    elif include_bedroom:
        from bedroom_categories import BEDROOM_CATEGORIES
        cats = list(EXTRA_HOUSE_CATEGORIES) + list(BEDROOM_CATEGORIES)
    else:
        cats = list(EXTRA_HOUSE_CATEGORIES)

    # de-dupe, preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    if force:
        return ordered

    have = local_categories(output_dir)
    missing = [c for c in ordered if c not in have]
    skipped = [c for c in ordered if c in have]
    if skipped:
        print(
            f"Skipping {len(skipped)} already-local categorie(s): "
            + ", ".join(skipped)
        )
    return missing


def build_arg_parser() -> argparse.ArgumentParser:
    p = _base_parser()
    p.description = (
        "Download extra house-related OmniObject3D categories "
        "(kitchen, living room, bathroom, utility) beyond the original 18."
    )
    p.add_argument(
        "--include-bedroom",
        action="store_true",
        help="Also include the original 18 bedroom categories in the plan "
        "(still skipped locally unless --force).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the category already exists under output-dir.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the default extra-house category list and exit.",
    )
    # Default categories=None so we control selection after --list / skip logic.
    p.set_defaults(categories=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        print(f"{len(EXTRA_HOUSE_CATEGORIES)} extra house categories:")
        print(" ".join(EXTRA_HOUSE_CATEGORIES))
        return 0

    output_dir: Path = args.output_dir
    chosen = pick_categories(
        args.categories,
        output_dir,
        force=args.force,
        include_bedroom=args.include_bedroom,
    )
    if not chosen:
        print("Nothing to download — all selected categories already present.")
        return 0
    if len(chosen) > 40 and not args.dry_run:
        print(
            f"Planning {len(chosen)} categories (large download). "
            "Use --dry-run first or --categories to narrow."
        )

    args.categories = chosen
    return _run_base(args)


if __name__ == "__main__":
    sys.exit(main())
