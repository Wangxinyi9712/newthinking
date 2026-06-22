from __future__ import annotations

import os
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.engine.trainer import MeanTeacherTrainer


def _build_model(cfg):
    return HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"],
        channels=cfg.model.get("channels", [32, 64, 128, 256]),
        use_transformer=cfg.model.get("use_transformer", True),
    )


def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    seeds = cfg.train.get("seed", [0])

    # ✔ FIX: remove seed-level folder for checkpoint consistency
    base_out = cfg.log["out_dir"]

    for seed in seeds:

        set_seed(seed)

        run_dir = base_out  # ✔ unified path

        os.makedirs(run_dir, exist_ok=True)

        loaders = build_dataloaders(cfg.data)

        model = _build_model(cfg)

        trainer = MeanTeacherTrainer(model, cfg)

        trainer.fit(
            loaders=loaders,
            out_dir=run_dir
        )

        print(f"[DONE] seed={seed}, saved to {run_dir}")


if __name__ == "__main__":
    main()