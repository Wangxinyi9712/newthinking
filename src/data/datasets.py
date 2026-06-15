from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monai.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from .transforms import (
    get_train_transforms_3d,
    get_train_unlabeled_transforms_3d,
    get_val_transforms_3d,
    get_train_transforms_2d,
    get_train_unlabeled_transforms_2d,
    get_val_transforms_2d,
)


def load_split_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _normalize_image(image, iid="unknown"):
    if isinstance(image, dict):
        keys = ["t1n", "t1c", "t2w", "t2f"]
        return [image[k] for k in keys]
    if isinstance(image, (list, tuple)):
        return list(image)
    return image


def _normalize(items):
    out = []
    for x in items:
        x = dict(x)
        x["image"] = _normalize_image(x["image"])
        out.append(x)
    return out


def build_dataloaders(data_cfg: dict):
    dim = data_cfg.get("dim", "3d")
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    split_path = data_cfg.get("split_file") or data_cfg.get("split_json")
    if split_path is None:
        raise KeyError("data config must contain split_file")

    splits = load_split_json(split_path)

    if dim == "3d":
        train_t = get_train_transforms_3d()
        train_u_t = get_train_unlabeled_transforms_3d()
        val_t = get_val_transforms_3d()
    else:
        train_t = get_train_transforms_2d()
        train_u_t = get_train_unlabeled_transforms_2d()
        val_t = get_val_transforms_2d()

    sup = _normalize(splits["labeled_train"])
    unsup = _normalize(splits["unlabeled_train"])
    val = _normalize(splits["val"])

    # ⚠️ 关键：禁用MONAI Cache（避免你当前OOM / thread crash）
    sup_ds = Dataset(sup, transform=train_t)
    unsup_ds = Dataset(unsup, transform=train_u_t)
    val_ds = Dataset(val, transform=val_t)

    labeled_loader = DataLoader(
        sup_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        unsup_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"[Data] labeled={len(sup)} unlabeled={len(unsup)} val={len(val)}")

    return {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader,
        "val": val_loader,
    }
