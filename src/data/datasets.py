from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monai.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from src.data.transforms import (
    get_train_transforms_2d,
    get_train_transforms_3d,
    get_train_unlabeled_transforms_2d,
    get_train_unlabeled_transforms_3d,
    get_val_transforms_2d,
    get_val_transforms_3d,
)


def load_split_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n[Split file not found]\n"
            f"Expected split file: {path}\n"
            f"Current working directory may be wrong.\n"
            f"Please run training from project root, for example:\n"
            f"  cd D:\\Code\\newthinking\n"
            f"  python src\\scripts\\train.py\n"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Split file must be a JSON object, got {type(data)}")

    return data


def _pick_split_list(splits: dict[str, Any], candidates: list[str], name: str) -> list[dict[str, Any]]:
    for key in candidates:
        if key in splits:
            value = splits[key]
            if value is None:
                return []
            if not isinstance(value, list):
                raise TypeError(f"Split key '{key}' should be a list, got {type(value)}")
            return value

    available = ", ".join(splits.keys())
    raise KeyError(
        f"\n[Split key missing]\n"
        f"Cannot find {name} split.\n"
        f"Tried keys: {candidates}\n"
        f"Available keys in split json: {available}\n"
    )


def _normalize_image(image: Any, item_id: str = "unknown") -> str | list[str]:
    """
    BraTS image should be:
        {"t1n": "...", "t1c": "...", "t2w": "...", "t2f": "..."}
    It will be converted to a 4-modality list in fixed order.
    """
    if isinstance(image, dict):
        order = ["t1n", "t1c", "t2w", "t2f"]
        missing = [k for k in order if k not in image or not image[k]]

        if missing:
            raise ValueError(
                f"\n[Invalid BraTS item]\n"
                f"item_id={item_id}\n"
                f"Missing modalities: {missing}\n"
                f"Expected keys: {order}\n"
                f"Actual keys: {list(image.keys())}\n"
            )

        return [image[k] for k in order]

    if isinstance(image, tuple):
        return list(image)

    if isinstance(image, list):
        return image

    if isinstance(image, str):
        return image

    raise TypeError(
        f"\n[Unsupported image field]\n"
        f"item_id={item_id}\n"
        f"type={type(image)}\n"
        f"value={image}\n"
    )


def _normalize_items(items: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise TypeError(f"{split_name}[{idx}] should be dict, got {type(raw)}")

        item = dict(raw)
        item_id = str(item.get("id", item.get("case_id", idx)))

        if "image" not in item:
            raise KeyError(
                f"\n[Invalid split item]\n"
                f"split={split_name}\n"
                f"item_id={item_id}\n"
                f"Missing key: image\n"
                f"Available keys: {list(item.keys())}\n"
            )

        item["image"] = _normalize_image(item["image"], item_id=item_id)
        out.append(item)

    return out


def _strip_unlabeled(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Unlabeled samples should not contain label during transform.
    """
    out: list[dict[str, Any]] = []

    for raw in items:
        item = {"image": raw["image"]}

        if "id" in raw:
            item["id"] = raw["id"]

        if "case_id" in raw:
            item["case_id"] = raw["case_id"]

        if "minority_score" in raw:
            item["minority_score"] = raw["minority_score"]

        out.append(item)

    return out


def _validate_non_empty(items: list[dict[str, Any]], split_name: str, split_path: str | Path) -> None:
    if len(items) == 0:
        raise ValueError(
            f"\n[Empty split error]\n"
            f"Split '{split_name}' has 0 samples.\n"
            f"Split file: {split_path}\n\n"
            f"This is the real reason for:\n"
            f"  ValueError: num_samples should be a positive integer value, but got num_samples=0\n\n"
            f"Please check your split json. It should contain non-empty lists, for example:\n"
            f"  labeled_train: [{{'image': ..., 'label': ...}}, ...]\n"
            f"  unlabeled_train: [{{'image': ...}}, ...]\n"
            f"  val: [{{'image': ..., 'label': ...}}, ...]\n\n"
            f"If you are on Windows, also check that the image/label paths in the split json are valid Windows paths.\n"
        )


def _validate_labeled_has_label(items: list[dict[str, Any]], split_name: str) -> None:
    bad = [i for i, x in enumerate(items[:10]) if "label" not in x]

    if bad:
        raise KeyError(
            f"\n[Invalid labeled split]\n"
            f"Split '{split_name}' contains samples without 'label'.\n"
            f"Bad sample indices among first 10: {bad}\n"
        )


def _build_minority_sampler(
    items: list[dict[str, Any]],
    enabled: bool,
    power: float = 1.5,
):
    if not enabled:
        return None

    if len(items) == 0:
        return None

    if not any("minority_score" in x for x in items):
        print("[Data] minority_oversample=True but no minority_score found; fallback to shuffle=True.")
        return None

    weights = []
    for item in items:
        score = float(item.get("minority_score", 0.0))
        weights.append((1e-3 + score) ** power)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(data_cfg: dict[str, Any]) -> dict[str, DataLoader]:
    dim = str(data_cfg.get("dim", "3d")).lower()

    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    spatial_size = data_cfg.get(
        "spatial_size",
        data_cfg.get("crop_size", [96, 96, 96] if dim == "3d" else [256, 256]),
    )

    split_path = data_cfg.get("split_file", data_cfg.get("split_json"))
    if split_path is None:
        raise KeyError("data config must contain 'split_file' or 'split_json'.")

    splits = load_split_json(split_path)

    raw_sup = _pick_split_list(
        splits,
        candidates=["labeled_train", "train_labeled", "labeled", "train"],
        name="labeled training",
    )

    raw_unsup = _pick_split_list(
        splits,
        candidates=["unlabeled_train", "train_unlabeled", "unlabeled", "unsupervised"],
        name="unlabeled training",
    )

    raw_val = _pick_split_list(
        splits,
        candidates=["val", "validation", "valid", "test"],
        name="validation",
    )

    sup_items = _normalize_items(raw_sup, split_name="labeled_train")
    unsup_items = _strip_unlabeled(_normalize_items(raw_unsup, split_name="unlabeled_train"))
    val_items = _normalize_items(raw_val, split_name="val")

    _validate_non_empty(sup_items, "labeled_train", split_path)
    _validate_non_empty(unsup_items, "unlabeled_train", split_path)
    _validate_non_empty(val_items, "val", split_path)

    _validate_labeled_has_label(sup_items, "labeled_train")
    _validate_labeled_has_label(val_items, "val")

    if dim == "3d":
        train_t = get_train_transforms_3d(spatial_size)
        train_u_t = get_train_unlabeled_transforms_3d(spatial_size)
        val_t = get_val_transforms_3d(spatial_size)
    elif dim == "2d":
        train_t = get_train_transforms_2d(spatial_size)
        train_u_t = get_train_unlabeled_transforms_2d(spatial_size)
        val_t = get_val_transforms_2d(spatial_size)
    else:
        raise ValueError(f"Unsupported data dim: {dim}. Use '2d' or '3d'.")

    sup_ds = Dataset(data=sup_items, transform=train_t)
    unsup_ds = Dataset(data=unsup_items, transform=train_u_t)
    val_ds = Dataset(data=val_items, transform=val_t)

    sampler = _build_minority_sampler(
        sup_items,
        enabled=bool(data_cfg.get("minority_oversample", False)),
        power=float(data_cfg.get("minority_oversample_power", 1.5)),
    )

    labeled_loader = DataLoader(
        sup_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
    )

    unlabeled_loader = DataLoader(
        unsup_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
    )

    print(
        f"[Data] split={split_path} | dim={dim} | spatial_size={list(spatial_size)} | "
        f"labeled={len(sup_items)} unlabeled={len(unsup_items)} val={len(val_items)} | "
        f"batch={batch_size} workers={num_workers}"
    )

    first = sup_items[0]
    print(f"[Data] first labeled keys={list(first.keys())}")

    if isinstance(first.get("image"), list):
        print(f"[Data] first labeled modalities={len(first['image'])}")

    return {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader,
        "val": val_loader,
    }