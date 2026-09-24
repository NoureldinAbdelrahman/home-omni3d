"""Bedroom-themed subset of OmniObject3D categories.

OmniObject3D has no literal ``lamp`` / ``nightstand`` / ``wardrobe`` labels;
the closest official categories are aliased below so directory names stay
aligned with the remote tarball / HDF5 filenames.
"""

from __future__ import annotations

__all__ = [
    "BEDROOM_CATEGORIES",
    "CATEGORY_ALIASES",
    "resolve_category",
]

# Official OmniObject3D category folder names (remote filenames).
BEDROOM_CATEGORIES: list[str] = [
    "bed",
    "pillow",
    "chair",
    "light",  # lamp / ceiling light
    "cabinet",  # wardrobe, nightstand-style storage
    "table",  # nightstand / side table
    "sofa",
    "stool",
    "clock",  # alarm / wall clock
    "tvstand",
    "vase",
    "tissue",
    "teddy_bear",
    "doll",
    "plant",
    "fan",
    "suitcase",
    "hair_dryer",
]

# Friendly name -> official category (for docs / user-facing overrides).
CATEGORY_ALIASES: dict[str, str] = {
    "lamp": "light",
    "wardrobe": "cabinet",
    "nightstand": "table",
    "night_stand": "table",
}


def resolve_category(name: str) -> str:
    """Map a user-supplied label to an official OmniObject3D category name."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(key, key)
