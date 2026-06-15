from __future__ import annotations

import os
import random
import numpy as np
import torch

from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.engine.trainer import MeanTeacherTrainer
from src.models.seg_model import HybridUNet


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    # -------------------------
    # SAFE CFG ACCESS (FIX KEYERROR)
    # -------------------------
    train_cfg = cfg.get("train", {})
    seeds = train_cfg.get("seed", [0])
    seed = seeds[0] if isinstance(seeds, list) else int(seeds)

    seed_all(seed)

    # -------------------------
    # DATA
    # -------------------------
    loaders = build_dataloaders(cfg["data"])

    # -------------------------
    # MODEL
    # -------------------------
    model_cfg = cfg.get("model", {})
    model = HybridUNet(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("channels", [32, 64, 128, 256])[0],
    )

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = MeanTeacherTrainer(model, cfg)

    out_dir = cfg.get("log", {}).get("out_dir", "runs/exp")

    trainer.fit(loaders, out_dir)


if __name__ == "__main__":
    main()