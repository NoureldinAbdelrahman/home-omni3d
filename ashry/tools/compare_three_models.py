#!/usr/bin/env python3
"""Create held-out-view comparison sheets for 3DGS, NeRF, and NeuS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def load(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../benchmark/results/benchmark_v1.json"))
    parser.add_argument("--gs-results", type=Path, required=True)
    parser.add_argument("--nerf-results", type=Path, required=True)
    parser.add_argument("--neus-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include objects missing one or more model render sets.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["objects"] if record["split"] == "val"]
    records = records if args.limit < 0 else records[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Image.Image] = []
    for record in records:
        slug = f'{record["category"]}_{record["object_id"]}'
        roots = {
            "3DGS": args.gs_results / slug / "heldout",
            "NeRF": args.nerf_results / slug / "heldout",
            "NeuS": args.neus_results / slug / "heldout",
        }
        complete = all(
            all((root / name).is_file() for name in record["heldout_views"])
            for root in roots.values()
        )
        if not complete and not args.allow_incomplete:
            continue
        render_dir = Path(manifest["dataset_dir"]) / record["render_dir"]
        tiles: list[tuple[str, Image.Image]] = []
        for name in record["heldout_views"]:
            tiles.append(("GT " + name, load(render_dir / name, args.tile_size)))
            for label, root in roots.items():
                path = root / name
                if path.is_file():
                    tiles.append((label + " " + name, load(path, args.tile_size)))
        if not tiles:
            continue
        columns = 4
        label_height = 24
        sheet = Image.new(
            "RGB",
            (columns * args.tile_size, ((len(tiles) + columns - 1) // columns) * (args.tile_size + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (label, tile) in enumerate(tiles):
            x = (index % columns) * args.tile_size
            y = (index // columns) * (args.tile_size + label_height)
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + args.tile_size + 3), label, fill="black")
        sheet.save(args.output_dir / f"{slug}.png")
        sheets.append(sheet.copy())

    if sheets:
        columns = 2
        width = columns * sheets[0].width
        rows = (len(sheets) + columns - 1) // columns
        contact = Image.new("RGB", (width, rows * sheets[0].height), "white")
        for index, sheet in enumerate(sheets):
            contact.paste(sheet, ((index % columns) * sheets[0].width, (index // columns) * sheets[0].height))
        contact.save(args.output_dir / "contact_sheet.png")
    print(f"Wrote {len(sheets)} comparison sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
