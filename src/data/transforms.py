from __future__ import annotations

from typing import Sequence

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
)


def get_train_transforms_3d(spatial_size: Sequence[int] = (128, 128, 128)):
    """
    3D 有标注训练变换
    关键点：
    - 不使用 Orientationd，避免 axcodes=3D 与 4D(含通道)冲突
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=tuple(spatial_size),
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_train_unlabeled_transforms_3d(spatial_size: Sequence[int] = (128, 128, 128)):
    """
    3D 无标注训练变换（只处理 image）
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandSpatialCropd(
                keys=["image"],
                roi_size=tuple(spatial_size),
                random_size=False,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.3),
            EnsureTyped(keys=["image"]),
        ]
    )


def get_val_transforms_3d(spatial_size: Sequence[int] = (128, 128, 128)):
    """
    3D 验证变换
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=tuple(spatial_size)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_train_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    """
    2D 有标注训练变换
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=tuple(spatial_size),
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.2,),
                scale_range=(0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_train_unlabeled_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    """
    2D 无标注训练变换
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            RandSpatialCropd(
                keys=["image"],
                roi_size=tuple(spatial_size),
                random_size=False,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.3),
            EnsureTyped(keys=["image"]),
        ]
    )


def get_val_transforms_2d(spatial_size: Sequence[int] = (256, 256)):
    """
    2D 验证变换
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=tuple(spatial_size)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )