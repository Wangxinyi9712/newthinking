from monai.transforms import *

# ---------------- 3D ----------------

def get_train_transforms_3d():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_unlabeled_transforms_3d():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
        EnsureTyped(keys=["image"]),
    ])


def get_val_transforms_3d():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])


# ---------------- 2D (新增) ----------------

def get_train_transforms_2d():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeD(keys=["image", "label"], spatial_size=(256, 256)),
        NormalizeIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_unlabeled_transforms_2d():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ResizeD(keys=["image"], spatial_size=(256, 256)),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])


def get_val_transforms_2d():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeD(keys=["image", "label"], spatial_size=(256, 256)),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])