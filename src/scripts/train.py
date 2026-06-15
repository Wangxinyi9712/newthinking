from __future__ import annotations

import random
import numpy as np
import torch

from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.engine.trainer import MeanTeacherTrainer
from src.models.seg_model import HybridUNet, ModelWrapper


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    train_cfg = cfg.get("train", {})
    seed = train_cfg.get("seed", [0])[0]
    seed_all(seed)

    loaders = build_dataloaders(cfg["data"])

    model_cfg = cfg.get("model", {})

    # =====================================================
    # FIX: NO base_channels (remove broken assumption)
    # =====================================================
    model = HybridUNet(
        in_channels=model_cfg.get("in_channels", 4),
        out_channels=model_cfg.get("out_channels", 1),
    )

    # =====================================================
    # IMPORTANT: unify interface
    # =====================================================
    model = ModelWrapper(model)

    trainer = MeanTeacherTrainer(model, cfg)

    trainer.fit(
        loaders,
        cfg.get("log", {}).get("out_dir", "runs/exp")
    )


if __name__ == "__main__":
    main()