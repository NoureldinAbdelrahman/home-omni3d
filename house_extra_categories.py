"""Extra house-themed OmniObject3D categories beyond the original 18 bedroom set.

Official remote folder names (must match OpenDataLab tarball / HDF5 filenames).
These cover kitchen, living-room electronics, bathroom, entryway, and tools
so an initial training run can start before the full bedroom dump is copied over.
"""

from __future__ import annotations

__all__ = [
    "EXTRA_HOUSE_CATEGORIES",
    "ALL_HOUSE_CATEGORIES",
]

# Kitchen / dining
# Living room / desk electronics
# Bathroom / personal care
# Entryway / utility / tools
EXTRA_HOUSE_CATEGORIES: list[str] = [
    # kitchen
    "kettle",
    "microwaveoven",
    "ricecooker",
    "pan",
    "dish",
    "cup",
    "bowl",
    "bottle",
    "teapot",
    "thermos",
    "timer",
    # living room / desk
    "remote_control",
    "speaker",
    "projector",
    "keyboard",
    "laptop",
    "monitor",
    "mouse",
    "power_strip",
    "plug",
    # bathroom / personal care
    "shampoo",
    "soap",
    "tooth_brush",
    "tooth_paste",
    "razor",
    "medicine_bottle",
    # entryway / utility / tools
    "dustbin",
    "fire_extinguisher",
    "flash_light",
    "hammer",
    "scissor",
    "umbrella",
]

# Original bedroom set lives in bedroom_categories.BEDROOM_CATEGORIES.
# Import lazily-safe: keep a local copy so this module has no hard dependency.
_BEDROOM_FALLBACK = [
    "bed", "pillow", "chair", "light", "cabinet", "table", "sofa", "stool",
    "clock", "tvstand", "vase", "tissue", "teddy_bear", "doll", "plant",
    "fan", "suitcase", "hair_dryer",
]


def _bedroom() -> list[str]:
    try:
        from bedroom_categories import BEDROOM_CATEGORIES
        return list(BEDROOM_CATEGORIES)
    except Exception:
        return list(_BEDROOM_FALLBACK)


ALL_HOUSE_CATEGORIES: list[str] = _bedroom() + EXTRA_HOUSE_CATEGORIES
