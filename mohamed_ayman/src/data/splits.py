"""
Stratified splits generator for OmniObject3D dataset.
Preserves category distributions across train, val, and test splits,
with dedicated tracking of Home Decoration & Furniture classes.
"""

import os
import json
import random
from typing import Dict, List, Tuple

HOME_DECOR_CATEGORIES = [
    "bed", "chair", "sofa", "table", "stool", "cabinet", "tvstand",
    "pillow", "light", "vase", "clock", "plant", "ornaments",
    "candle", "dish", "cup", "bowl", "teapot"
]


def find_dataset_root(custom_root: str = None) -> str:
    if custom_root and os.path.isdir(os.path.join(custom_root, "renders")):
        return custom_root
    candidates = [
        custom_root,
        "dataset",
        "../dataset",
        "../../dataset",
        os.path.join(os.path.dirname(__file__), "..", "..", "dataset"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "renders")):
            return c
    return "dataset"


def create_or_load_splits(
    dataset_root: str = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_path: str = None
) -> Dict[str, List[Dict[str, str]]]:
    """
    Creates or loads stratified object-level splits.
    Each sample is a dict: {"category": cat, "object_id": obj_id}
    """
    dataset_root = find_dataset_root(dataset_root)
    if save_path is None:
        save_path = os.path.join(dataset_root, "splits.json")

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)

    renders_dir = os.path.join(dataset_root, "renders")
    pcs_dir = os.path.join(dataset_root, "point_clouds")

    categories = sorted([
        d for d in os.listdir(renders_dir)
        if os.path.isdir(os.path.join(renders_dir, d))
    ])

    rng = random.Random(seed)
    train_samples: List[Dict[str, str]] = []
    val_samples: List[Dict[str, str]] = []
    test_samples: List[Dict[str, str]] = []

    for cat in categories:
        cat_render_dir = os.path.join(renders_dir, cat)
        cat_pc_dir = os.path.join(pcs_dir, cat)

        # List valid objects having both render folder and .npy point cloud
        objs = sorted([
            o for o in os.listdir(cat_render_dir)
            if os.path.isdir(os.path.join(cat_render_dir, o)) and
            os.path.exists(os.path.join(cat_pc_dir, f"{o}.npy"))
        ])

        if not objs:
            continue

        rng.shuffle(objs)
        n = len(objs)

        if n == 1:
            train_samples.append({"category": cat, "object_id": objs[0]})
        elif n == 2:
            train_samples.append({"category": cat, "object_id": objs[0]})
            test_samples.append({"category": cat, "object_id": objs[1]})
        elif n == 3:
            train_samples.append({"category": cat, "object_id": objs[0]})
            val_samples.append({"category": cat, "object_id": objs[1]})
            test_samples.append({"category": cat, "object_id": objs[2]})
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
            n_test = n - n_train - n_val

            train_objs = objs[:n_train]
            val_objs = objs[n_train:n_train + n_val]
            test_objs = objs[n_train + n_val:]

            for o in train_objs:
                train_samples.append({"category": cat, "object_id": o})
            for o in val_objs:
                val_samples.append({"category": cat, "object_id": o})
            for o in test_objs:
                test_samples.append({"category": cat, "object_id": o})

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "home_decor_categories": HOME_DECOR_CATEGORIES,
        "all_categories": categories
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created dataset splits: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test.")
    return splits


if __name__ == "__main__":
    create_or_load_splits()
