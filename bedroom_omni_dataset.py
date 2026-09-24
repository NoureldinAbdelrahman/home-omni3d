"""PyTorch dataset over the local bedroom OmniObject3D layout.

Expected directory structure (produced by ``download_bedroom_subset.py``)::

    dataset/
    ├── point_clouds/
    │   ├── bed/
    │   │   └── bed_001.npy      # (num_points, 3) float32
    │   └── light/
    └── renders/
        ├── bed/
        │   └── bed_001/
        │       ├── 000.png ... 023.png
        │       └── transforms.json
        └── light/
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Modality = Literal["renders", "point_clouds", "both"]
ViewMode = Literal["first", "random", "all"]

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class BedroomOmniDataset(Dataset):
    """Bedroom-subset OmniObject3D loader for multi-view renders + point clouds.

    Parameters
    ----------
    root:
        Dataset root containing ``renders/`` and/or ``point_clouds/``.
    modality:
        ``"renders"``, ``"point_clouds"``, or ``"both"``.
    num_points:
        Target point count. ``None`` keeps the stored resolution (point
        clouds are still randomly subsampled / zero-padded to a fixed size
        only when this is an ``int``).
    view_mode:
        How to expose the 24 Blender views: return a single ``"first"``
        image, a ``"random"`` view per ``__getitem__`` call, or a stacked
        ``"all"`` tensor of shape ``(V, 3, H, W)``.
    image_size:
        Optional ``(H, W)`` resize. ``None`` keeps native resolution
        (native renders are square).
    normalize:
        If ``True``, scale pixels to ``[0, 1]`` and apply ImageNet stats.
    include_camera:
        If ``True``, also return the raw ``transforms.json`` dict per sample.
        Off by default — nested dicts collate poorly across a batch.
    class_names:
        Optional explicit class list; defaults to sorted categories found
        on disk.
    transform / target_transform:
        Callables applied to the sample / label.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        modality: Modality = "both",
        num_points: int | None = 1024,
        view_mode: ViewMode = "first",
        image_size: int | tuple[int, int] | None = None,
        normalize: bool = True,
        include_camera: bool = False,
        class_names: Sequence[str] | None = None,
        require_renders: bool = True,
        require_point_clouds: bool = True,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], int] | None = None,
    ) -> None:
        super().__init__()
        if modality not in ("renders", "point_clouds", "both"):
            raise ValueError(f"Invalid modality: {modality!r}")
        if view_mode not in ("first", "random", "all"):
            raise ValueError(f"Invalid view_mode: {view_mode!r}")

        self.root = Path(root)
        self.modality = modality
        self.num_points = num_points
        self.view_mode = view_mode
        self.include_camera = include_camera
        self.image_size = (
            image_size
            if image_size is None or isinstance(image_size, tuple)
            else (image_size, image_size)
        )
        self.normalize = normalize
        self.transform = transform
        self.target_transform = target_transform

        self.renders_root = self.root / "renders"
        self.point_clouds_root = self.root / "point_clouds"

        need_renders = modality in ("renders", "both")
        need_pcs = modality in ("point_clouds", "both")
        if need_renders and require_renders and not self.renders_root.is_dir():
            raise FileNotFoundError(
                f"Renders directory not found: {self.renders_root}. "
                "Run download_bedroom_subset.py first."
            )
        if need_pcs and require_point_clouds and not self.point_clouds_root.is_dir():
            raise FileNotFoundError(
                f"Point cloud directory not found: {self.point_clouds_root}. "
                "Run download_bedroom_subset.py first."
            )

        self.samples: list[dict[str, Any]] = []
        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []
        self._index(need_renders, need_pcs, class_names)

        # ImageNet normalization buffers (lazy; registered as plain attrs so
        # the Dataset stays picklable for DataLoader workers).
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def _index(
        self,
        need_renders: bool,
        need_pcs: bool,
        class_names: Sequence[str] | None,
    ) -> None:
        if class_names is None:
            names: set[str] = set()
            if need_renders and self.renders_root.is_dir():
                names |= {p.name for p in self.renders_root.iterdir() if p.is_dir()}
            if need_pcs and self.point_clouds_root.is_dir():
                names |= {p.name for p in self.point_clouds_root.iterdir() if p.is_dir()}
            class_names = sorted(names)
        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Build object-id universe from whichever modality is present.
        object_ids: dict[str, set[str]] = {c: set() for c in self.classes}

        render_index: dict[tuple[str, str], Path] = {}
        if need_renders and self.renders_root.is_dir():
            for cls in self.classes:
                cls_dir = self.renders_root / cls
                if not cls_dir.is_dir():
                    continue
                for obj_dir in sorted(cls_dir.iterdir()):
                    if not obj_dir.is_dir():
                        continue
                    if not any(p.suffix.lower() in IMG_EXTENSIONS for p in obj_dir.iterdir()):
                        continue
                    render_index[(cls, obj_dir.name)] = obj_dir
                    object_ids[cls].add(obj_dir.name)

        pc_index: dict[tuple[str, str], Path] = {}
        if need_pcs and self.point_clouds_root.is_dir():
            for cls in self.classes:
                cls_dir = self.point_clouds_root / cls
                if not cls_dir.is_dir():
                    continue
                for pc_file in sorted(cls_dir.glob("*.npy")):
                    object_id = pc_file.stem
                    pc_index[(cls, object_id)] = pc_file
                    object_ids[cls].add(object_id)

        for cls in self.classes:
            if not object_ids[cls]:
                continue
            for object_id in sorted(object_ids[cls]):
                render_path = render_index.get((cls, object_id))
                pc_path = pc_index.get((cls, object_id))
                if need_renders and render_path is None:
                    continue
                if need_pcs and pc_path is None:
                    continue
                self.samples.append(
                    {
                        "category": cls,
                        "label": self.class_to_idx[cls],
                        "object_id": object_id,
                        "render_dir": render_path,
                        "point_cloud": pc_path,
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"No usable samples under {self.root} for modality={self.modality!r}"
            )

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    def _list_views(self, render_dir: Path) -> list[Path]:
        views = sorted(
            p for p in render_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not views:
            raise FileNotFoundError(f"No images in {render_dir}")
        return views

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.image_size is not None and img.size != (
                self.image_size[1],
                self.image_size[0],
            ):
                img = img.resize(
                    (self.image_size[1], self.image_size[0]), Image.BILINEAR
                )
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.normalize:
            tensor = (tensor - self._mean) / self._std
        return tensor

    def _load_views(self, render_dir: Path) -> torch.Tensor:
        paths = self._list_views(render_dir)
        if self.view_mode == "first":
            return self._load_image(paths[0])
        if self.view_mode == "random":
            return self._load_image(random.choice(paths))
        return torch.stack([self._load_image(p) for p in paths], dim=0)

    def _load_point_cloud(self, path: Path) -> torch.Tensor:
        cloud = np.load(path).astype(np.float32, copy=False)
        if cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError(f"Expected (N, >=3) point cloud at {path}, got {cloud.shape}")
        cloud = cloud[:, :3]
        tensor = torch.from_numpy(np.ascontiguousarray(cloud))

        if self.num_points is not None:
            n = tensor.shape[0]
            if n >= self.num_points:
                idx = torch.randperm(n)[: self.num_points]
                tensor = tensor[idx]
            else:
                pad = torch.zeros(self.num_points - n, 3, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
        return tensor

    def _load_camera(self, render_dir: Path) -> dict[str, Any] | None:
        meta = render_dir / "transforms.json"
        if not meta.is_file():
            return None
        with meta.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        label = sample["label"]
        out: dict[str, Any] = {
            "object_id": sample["object_id"],
            "category": sample["category"],
            "label": label,
        }

        if self.modality in ("renders", "both"):
            out["image"] = self._load_views(sample["render_dir"])
            if self.include_camera:
                camera = self._load_camera(sample["render_dir"])
                if camera is not None:
                    out["camera"] = camera

        if self.modality in ("point_clouds", "both"):
            out["point_cloud"] = self._load_point_cloud(sample["point_cloud"])

        if self.transform is not None:
            out = self.transform(out)
        if self.target_transform is not None:
            out["label"] = self.target_transform(label)
        return out

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, "
            f"samples={len(self)}, classes={len(self.classes)}, "
            f"modality={self.modality!r}, num_points={self.num_points}, "
            f"view_mode={self.view_mode!r})"
        )


def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test BedroomOmniDataset")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--modality", default="both", choices=("renders", "point_clouds", "both"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = BedroomOmniDataset(
        args.root, modality=args.modality, num_points=args.num_points  # type: ignore[arg-type]
    )
    print(ds)
    sample = ds[args.index]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor{tuple(value.shape)} {value.dtype}")
        elif key == "camera":
            print(f"  {key}: dict with {len(value)} keys")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    _demo()
