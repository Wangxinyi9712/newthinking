from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _write_split(path: str | Path, split: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)


def _find_first_existing(case_dir: Path, names: list[str]) -> str | None:
    for n in names:
        p = case_dir / n
        if p.exists():
            # 过滤 macOS 资源叉文件
            if p.name.startswith("._"):
                continue
            return str(p)
    return None


def _is_valid_modality_pack(img_dict: dict[str, str | None], min_modalities: int = 4) -> bool:
    cnt = sum(v is not None for v in img_dict.values())
    return cnt >= min_modalities


def _split_items(
    items: list[dict[str, Any]],
    val_ratio: float,
    labeled_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    if not (0.0 < labeled_ratio < 1.0):
        raise ValueError(f"labeled_ratio must be in (0,1), got {labeled_ratio}")
    if val_ratio + labeled_ratio >= 1.0:
        raise ValueError("val_ratio + labeled_ratio must be < 1.0")

    n = len(items)
    if n == 0:
        return {"labeled_train": [], "unlabeled_train": [], "val": []}

    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)

    n_val = max(1, int(round(n * val_ratio)))
    n_labeled = max(1, int(round(n * labeled_ratio)))
    if n_val + n_labeled > n:
        n_labeled = max(1, n - n_val)

    val_items = items[:n_val]
    train_items = items[n_val:]
    labeled_items = train_items[:n_labeled]
    unlabeled_items = train_items[n_labeled:]

    return {
        "labeled_train": labeled_items,
        "unlabeled_train": unlabeled_items,
        "val": val_items,
    }


def build_brats_split(
    brats_root: str | Path,
    out_json: str | Path,
    val_ratio: float = 0.10,
    labeled_ratio: float = 0.10,
    seed: int = 3407,
) -> None:
    root = Path(brats_root)
    train_root = root / "training_data" if (root / "training_data").exists() else root
    if not train_root.exists():
        raise FileNotFoundError(f"BraTS train root not found: {train_root}")

    cases = sorted([p for p in train_root.iterdir() if p.is_dir() and not p.name.startswith("._")])
    all_items: list[dict[str, Any]] = []

    dropped_no_seg = 0
    dropped_incomplete_modalities = 0

    for case in cases:
        cid = case.name
        if cid.startswith("._"):
            continue

        t1n = _find_first_existing(case, [f"{cid}-t1n.nii.gz", "t1n.nii.gz"])
        t1c = _find_first_existing(case, [f"{cid}-t1c.nii.gz", "t1c.nii.gz"])
        t2w = _find_first_existing(case, [f"{cid}-t2w.nii.gz", "t2w.nii.gz"])
        t2f = _find_first_existing(case, [f"{cid}-t2f.nii.gz", "t2f.nii.gz"])
        seg = _find_first_existing(case, [f"{cid}-seg.nii.gz", "seg.nii.gz", "label.nii.gz"])

        image_dict = {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}

        if seg is None:
            dropped_no_seg += 1
            continue

        # 关键：论文可比性要求 4 模态齐全
        if not _is_valid_modality_pack(image_dict, min_modalities=4):
            dropped_incomplete_modalities += 1
            continue

        all_items.append(
            {
                "id": cid,
                "image": image_dict,
                "label": seg,
                "minority_score": 0.10,
            }
        )

    split = _split_items(all_items, val_ratio=val_ratio, labeled_ratio=labeled_ratio, seed=seed)
    _write_split(out_json, split)

    print(
        f"[BraTS split done] valid_cases={len(all_items)}, "
        f"dropped_no_seg={dropped_no_seg}, "
        f"dropped_incomplete_modalities={dropped_incomplete_modalities}, "
        f"labeled={len(split['labeled_train'])}, "
        f"unlabeled={len(split['unlabeled_train'])}, "
        f"val={len(split['val'])}, seed={seed}, out={out_json}"
    )


def build_isic_split(
    isic_root: str | Path,
    out_json: str | Path,
    val_ratio: float = 0.10,
    labeled_ratio: float = 0.10,
    seed: int = 3407,
) -> None:
    root = Path(isic_root)

    if (root / "images").exists() and (root / "masks").exists():
        images_dir = root / "images"
        masks_dir = root / "masks"
        mask_suffix = "_segmentation.png"
    else:
        images_dir = root / "ISIC2018_Task1-2_Training_Input"
        masks_dir = root / "ISIC2018_Task1_Training_GroundTruth"
        mask_suffix = "_segmentation.png"

    if not images_dir.exists():
        raise FileNotFoundError(f"ISIC images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"ISIC masks dir not found: {masks_dir}")

    all_items: list[dict[str, Any]] = []
    images = sorted([p for p in images_dir.glob("*.jpg") if not p.name.startswith("._")])

    for img in images:
        mask = masks_dir / f"{img.stem}{mask_suffix}"
        if not mask.exists() or mask.name.startswith("._"):
            continue
        all_items.append(
            {
                "id": img.stem,
                "image": str(img),
                "label": str(mask),
                "minority_score": 0.05,
            }
        )

    split = _split_items(all_items, val_ratio=val_ratio, labeled_ratio=labeled_ratio, seed=seed)
    _write_split(out_json, split)

    print(
        f"[ISIC split done] valid_images={len(all_items)}, "
        f"labeled={len(split['labeled_train'])}, "
        f"unlabeled={len(split['unlabeled_train'])}, "
        f"val={len(split['val'])}, seed={seed}, out={out_json}"
    )


def build_msd_liver_split(
    msd_liver_root: str | Path,
    out_json: str | Path,
    val_ratio: float = 0.10,
    labeled_ratio: float = 0.10,
    seed: int = 3407,
) -> None:
    root = Path(msd_liver_root)

    task_root = root / "Task03_Liver" if (root / "Task03_Liver").exists() else root
    images_tr = task_root / "imagesTr"
    labels_tr = task_root / "labelsTr"

    if not images_tr.exists():
        raise FileNotFoundError(f"MSD imagesTr dir not found: {images_tr}")
    if not labels_tr.exists():
        raise FileNotFoundError(f"MSD labelsTr dir not found: {labels_tr}")

    all_items: list[dict[str, Any]] = []
    imgs = sorted([p for p in images_tr.glob("*.nii*") if not p.name.startswith("._")])

    for img in imgs:
        label = labels_tr / img.name
        if not label.exists() or label.name.startswith("._"):
            continue

        stem = img.stem.replace(".nii", "")
        if stem.startswith("._"):
            continue

        all_items.append(
            {
                "id": stem,
                "image": str(img),
                "label": str(label),
                "minority_score": 0.08,
            }
        )

    split = _split_items(all_items, val_ratio=val_ratio, labeled_ratio=labeled_ratio, seed=seed)
    _write_split(out_json, split)

    print(
        f"[MSD Liver split done] valid_volumes={len(all_items)}, "
        f"labeled={len(split['labeled_train'])}, "
        f"unlabeled={len(split['unlabeled_train'])}, "
        f"val={len(split['val'])}, seed={seed}, out={out_json}"
    )