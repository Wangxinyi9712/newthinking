from __future__ import annotations

from typing import Sequence

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    SpatialPadd,
)


def _binarize_label(x):
    if hasattr(x, "astype"):
        return (x > 0).astype("float32")

    if isinstance(x, torch.Tensor):
        return (x > 0).float()

    return x


def get_train_transforms_3d(spatial_size: Sequence[int] = (96, 96, 96)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["label"], func=_binarize_label),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

            # 保证体数据至少不小于 crop size，避免 MSD/Liver 等数据出现裁剪错误。
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),

            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=spatial_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=(0.08, 0.08, 0.08),
                scale_range=(0.08, 0.08, 0.08),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_train_unlabeled_transforms_3d(spatial_size: Sequence[int] = (96, 96, 96)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

            SpatialPadd(keys=["image"], spatial_size=spatial_size),

            RandSpatialCropd(
                keys=["image"],
                roi_size=spatial_size,
                random_size=False,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.4),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.25),
            EnsureTyped(keys=["image"], track_meta=False),
        ]
    )


def get_val_transforms_3d(spatial_size: Sequence[int] = (96, 96, 96)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["label"], func=_binarize_label),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_train_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["label"], func=_binarize_label),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_train_unlabeled_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.4),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.25),
            EnsureTyped(keys=["image"], track_meta=False),
        ]
    )


def get_val_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    spatial_size = tuple(int(v) for v in spatial_size)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(keys=["label"], func=_binarize_label),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )