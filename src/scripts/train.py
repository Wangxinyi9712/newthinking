from __future__ import annotations

import os
import random
import numpy as np
import torch

from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.engine.trainer import MeanTeacherTrainer


# =========================
# reproducibility
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# config flatten helper
# =========================

def cfg_to_dict(cfg):
    """
    dataclass -> dict safe conversion
    """
    if hasattr(cfg, "__dict__"):
        return cfg.__dict__
    return cfg


# =========================
# main
# =========================

def main():

    cfg_obj = load_config("src/configs/brats_group_e.yaml")
    cfg = cfg_to_dict(cfg_obj)

    # seed handling
    seed_list = cfg["train"].get("seed", [0])
    set_seed(seed_list[0])

    print("[INFO] Loading dataset...")

    loaders = build_dataloaders(cfg["data"])

    print("[INFO] Building model...")

    model_cfg = cfg["model"]

    model = HybridUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg.get("channels", [32, 64, 128, 256]),
        use_transformer=model_cfg.get("use_transformer", True),
    )

    print("[INFO] Initializing trainer...")

    trainer = MeanTeacherTrainer(
        student=model,
        cfg=cfg
    )

    out_dir = cfg["log"]["out_dir"]

    print("[INFO] Start training...")

    trainer.fit(
        loaders=loaders,
        out_dir=out_dir
    )


if __name__ == "__main__":
    main()