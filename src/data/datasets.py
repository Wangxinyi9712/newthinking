import json
from pathlib import Path
from monai.data import Dataset, DataLoader
from src.data.transforms import get_train_transforms_3d, get_val_transforms_3d


def load_split_json(path):
    with open(path) as f:
        return json.load(f)


def build_dataloaders(cfg):

    splits = load_split_json(cfg["split_file"])

    train_t = get_train_transforms_3d(cfg.get("crop_size", [96,96,96]))
    val_t = get_val_transforms_3d(cfg.get("crop_size", [96,96,96]))

    train_ds = Dataset(splits["labeled_train"], transform=train_t)
    unlabeled_ds = Dataset(splits["unlabeled_train"], transform=train_t)
    val_ds = Dataset(splits["val"], transform=val_t)

    return {
        "labeled": DataLoader(train_ds, batch_size=1, shuffle=True),
        "unlabeled": DataLoader(unlabeled_ds, batch_size=1, shuffle=True),
        "val": DataLoader(val_ds, batch_size=1, shuffle=False),
    }