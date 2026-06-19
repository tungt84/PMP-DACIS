"""Local loader for PlantVillage dataset stored under data/plantvillage.

This is an independent implementation inspired by the official dataset script
`plant_village.py` but works locally without Hugging Face `datasets`.

Usage example:
from data.plantvillage.local_loader import load_local_dataset
splits = load_local_dataset("data/plantvillage", config="color")
print({k: len(v) for k, v in splits.items()})
"""

import os
import json
import random
from typing import Dict, List, Optional


def _parse_metadata_from_path(file_rel_path: str):
    parts = file_rel_path.split("/")
    if len(parts) < 4:
        return None
    # parts: raw/<type>/<class>/<filename>
    class_name = parts[2]
    file_name = parts[3]
    sub_parts = class_name.split("___")
    crop = sub_parts[0]
    disease = sub_parts[1] if len(sub_parts) > 1 else "unknown"
    return class_name, file_name, crop, disease


def _resolve_leaf_id(file_name: str, class_name: str, leaf_map: Dict[str, List[str]]):
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    image_identifier = image_identifier.replace(".jpg", "").replace(".JPG", "").replace(".png", "").replace(".PNG", "")
    image_identifier = image_identifier.strip()
    lookup_key = image_identifier.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        else:
            for suggestion in suggestions:
                if class_name in suggestion:
                    return suggestion
            return f"fallback_{image_identifier}"
    else:
        return f"fallback_{image_identifier}"


def _read_leaf_map(root: str) -> Dict[str, List[str]]:
    # leaf map path may be at root/leaf_grouping/leaf-map.json or root/leaf-map.json
    candidates = [
        os.path.join(root, "leaf_grouping", "leaf-map.json"),
        os.path.join(root, "leaf-map.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def _read_split_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_local_dataset(
    root: str,
    config: str = "color",
    use_splits_if_available: bool = True,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Load PlantVillage locally.

    Returns a dict with keys `train` and `test`, each a list of example dicts with keys:
    - image_path: relative path inside data zip (e.g. raw/color/Apple/123.jpg)
    - abs_path: absolute filesystem path
    - label: class name (e.g. Apple___healthy)
    - crop, disease, leaf_id

    If split files are present under `root/splits/{config}_train.txt` and `{config}_test.txt`, they are used.
    Otherwise the loader will build splits by grouping images by `leaf_id` (using available leaf-map) and
    performing an 80/20 split on leaf ids.
    """

    root = os.path.normpath(root)
    splits_dir = os.path.join(root, "splits")

    leaf_map = _read_leaf_map(root)

    # Try to read official split files
    train_list = None
    test_list = None
    if use_splits_if_available:
        train_path = os.path.join(splits_dir, f"{config}_train.txt")
        test_path = os.path.join(splits_dir, f"{config}_test.txt")
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_list = _read_split_file(train_path)
            test_list = _read_split_file(test_path)

    # If splits not available, enumerate all files under raw/<config>
    if train_list is None or test_list is None:
        raw_dir = os.path.join(root, "raw", config)
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        # Gather all file_rel_paths relative to root (so they match original script format)
        all_files = []
        for dirpath, _, filenames in os.walk(raw_dir):
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    abs_p = os.path.join(dirpath, fn)
                    rel_p = os.path.relpath(abs_p, root).replace(os.path.sep, "/")
                    all_files.append(rel_p)

        # Group by leaf_id using leaf_map if present, else fall back to filename-based grouping
        groups = {}
        for rel in all_files:
            meta = _parse_metadata_from_path(rel)
            if meta is None:
                continue
            class_name, file_name, _, _ = meta
            leaf_id = _resolve_leaf_id(file_name, class_name, leaf_map)
            groups.setdefault(leaf_id, []).append(rel)

        # Split leaf ids 80/20
        leaf_ids = list(groups.keys())
        random.Random(seed).shuffle(leaf_ids)
        split_idx = int(len(leaf_ids) * 0.8)
        train_leaf_ids = set(leaf_ids[:split_idx])
        test_leaf_ids = set(leaf_ids[split_idx:])

        train_list = []
        test_list = []
        for lid, files in groups.items():
            if lid in train_leaf_ids:
                train_list.extend(files)
            else:
                test_list.extend(files)

    # Build examples
    def build_examples(file_list: List[str]) -> List[Dict]:
        examples = []
        for rel in file_list:
            meta = _parse_metadata_from_path(rel)
            if meta is None:
                continue
            class_name, file_name, crop, disease = meta
            abs_path = os.path.join(root, rel)
            leaf_id = _resolve_leaf_id(file_name, class_name, leaf_map)
            examples.append(
                {
                    "image_path": rel,
                    "abs_path": abs_path,
                    "label": class_name,
                    "crop": crop,
                    "disease": disease,
                    "leaf_id": leaf_id,
                }
            )
        return examples

    train_examples = build_examples(train_list)
    test_examples = build_examples(test_list)

    return {"train": train_examples, "test": test_examples}
