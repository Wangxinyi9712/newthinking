from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monai.data import CacheDataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from .transforms import (
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


def _normalize_image_field(image: Any, item_id: str = "unknown") -> str | list[str]:
    """
    兼容:
    - BraTS: image 为 dict {t1n,t1c,t2w,t2f} -> 固定顺序 list[str]，且必须 4 模态齐全
    - 其他: image 为 str/list/tuple
    """
    if isinstance(image, dict):
        order = ["t1n", "t1c", "t2w", "t2f"]
        paths = [image.get(k) for k in order if image.get(k)]
        if len(paths) != 4:
            raise ValueError(
                f"[split] item={item_id} expects 4 modalities (t1n,t1c,t2w,t2f), got {len(paths)}. "
                f"raw_keys={list(image.keys())}"
            )
        return paths
    if isinstance(image, tuple):
        return list(image)
    if isinstance(image, list):
        return image
    if isinstance(image, str):
        return image
    raise TypeError(f"[split] item={item_id} unsupported image type: {type(image)}")


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for x in items:
        it = dict(x)
        iid = str(it.get("id", "unknown"))
        if "image" not in it:
            raise KeyError(f"[split] item={iid} missing key 'image'")
        it["image"] = _normalize_image_field(it["image"], item_id=iid)
        out.append(it)
    return out


def _build_minority_sampler(items: list[dict[str, Any]], enabled: bool, power: float = 2.0) -> WeightedRandomSampler | None:
    if not enabled:
        return None
    if any("minority_score" in x for x in items):
        weights = []
        for it in items:
            score = float(it.get("minority_score", 0.0))
            weights.append((1e-3 + score) ** power)
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return None


def _strip_unlabeled(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for x in items:
        item = {"image": x["image"]}
        if "minority_score" in x:
            item["minority_score"] = x["minority_score"]
        if "id" in x:
            item["id"] = x["id"]
        out.append(item)
    return out


def build_dataloaders(data_cfg: dict[str, Any]) -> dict[str, DataLoader]:
    dim = data_cfg.get("dim", "3d")
    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    cache_rate = float(data_cfg.get("cache_rate", 0.1))
    spatial_size = tuple(data_cfg.get("spatial_size", [128, 128, 128] if dim == "3d" else [256, 256]))

    # 兼容两种字段名：split_file / split_json
    split_path = data_cfg.get("split_file", data_cfg.get("split_json", None))
    if split_path is None:
        raise KeyError("data config must contain 'split_file' or 'split_json'.")

    splits = load_split_json(split_path)

    if dim == "3d":
        train_t = get_train_transforms_3d(spatial_size)
        train_u_t = get_train_unlabeled_transforms_3d(spatial_size)
        val_t = get_val_transforms_3d(spatial_size)
    else:
        train_t = get_train_transforms_2d(spatial_size)
        train_u_t = get_train_unlabeled_transforms_2d(spatial_size)
        val_t = get_val_transforms_2d(spatial_size)

    # 先标准化 image 字段（BraTS 强制 4 模态）
    sup_items = _normalize_items(splits.get("labeled_train", []))
    unsup_items = _normalize_items(splits.get("unlabeled_train", []))
    val_items = _normalize_items(splits.get("val", []))

    # 无标注数据去掉 label，避免 transform 误处理
    unsup_items = _strip_unlabeled(unsup_items)

    sup_ds = CacheDataset(sup_items, transform=train_t, cache_rate=cache_rate)
    unsup_ds = CacheDataset(unsup_items, transform=train_u_t, cache_rate=cache_rate)
    val_ds = CacheDataset(val_items, transform=val_t, cache_rate=cache_rate)

    sampler = _build_minority_sampler(
        sup_items,
        enabled=bool(data_cfg.get("minority_oversample", False)),
        power=float(data_cfg.get("minority_oversample_power", 2.0)),
    )

    safe_num_workers = num_workers
    safe_pin_memory = bool(data_cfg.get("pin_memory", False))

    labeled_loader = DataLoader(
        sup_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=safe_num_workers,
        pin_memory=safe_pin_memory,
        drop_last=True,
        persistent_workers=False,
    )

    unlabeled_loader = DataLoader(
        unsup_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_num_workers,
        pin_memory=safe_pin_memory,
        drop_last=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0 if safe_num_workers > 0 else safe_num_workers,
        pin_memory=safe_pin_memory,
        drop_last=False,
        persistent_workers=False,
    )

    print(
        f"[Data] split={split_path} | labeled={len(sup_items)} "
        f"unlabeled={len(unsup_items)} val={len(val_items)} | "
        f"batch={batch_size} workers={safe_num_workers} cache_rate={cache_rate} pin_memory={safe_pin_memory}"
    )
    if len(sup_items) > 0:
        t = sup_items[0]["image"]
        if isinstance(t, list):
            print(f"[Data] sample_labeled_image_type=list, modalities={len(t)}")
        else:
            print(f"[Data] sample_labeled_image_type={type(t).__name__}")

    return {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader,
        "val": val_loader,
    }