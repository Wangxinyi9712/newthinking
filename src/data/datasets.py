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


def load_split_json(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_image(image: Any, item_id: str = "unknown") -> str | list[str]:
    if isinstance(image, dict):
        order = ["t1n", "t1c", "t2w", "t2f"]
        missing = [k for k in order if k not in image or not image[k]]
        if missing:
            raise ValueError(f"[split] item={item_id} missing modalities: {missing}")
        return [image[k] for k in order]

    if isinstance(image, tuple):
        return list(image)

    if isinstance(image, list):
        return image

    if isinstance(image, str):
        return image

    raise TypeError(f"[split] item={item_id} unsupported image type: {type(image)}")


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for raw in items:
        item = dict(raw)
        item_id = str(item.get("id", "unknown"))

        if "image" not in item:
            raise KeyError(f"[split] item={item_id} missing key 'image'")

        item["image"] = _normalize_image(item["image"], item_id=item_id)
        out.append(item)

    return out


def _strip_unlabeled(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for raw in items:
        item = {"image": raw["image"]}

        if "id" in raw:
            item["id"] = raw["id"]

        if "minority_score" in raw:
            item["minority_score"] = raw["minority_score"]

        out.append(item)

    return out


def _build_minority_sampler(
    items: list[dict[str, Any]],
    enabled: bool,
    power: float = 1.5,
):
    if not enabled:
        return None

    if not any("minority_score" in x for x in items):
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

    sup_items = _normalize_items(splits.get("labeled_train", []))
    unsup_items = _strip_unlabeled(_normalize_items(splits.get("unlabeled_train", [])))
    val_items = _normalize_items(splits.get("val", []))

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

    if sup_items:
        sample_image = sup_items[0]["image"]
        if isinstance(sample_image, list):
            print(f"[Data] sample modalities={len(sample_image)}")
        else:
            print(f"[Data] sample image type={type(sample_image).__name__}")

    return {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader,
        "val": val_loader,
    }