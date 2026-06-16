from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.engine.trainer import MeanTeacherTrainer


# =========================================================
# Model builder (STRICT MATCH to current HybridUNet)
# =========================================================
def _build_model(cfg):
    """
    IMPORTANT:
    HybridUNet current signature ONLY supports:
        - in_channels
        - out_channels
    """
    return HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"],
    )


# =========================================================
# Main training entry
# =========================================================
def main():

    # -------------------------
    # config
    # -------------------------
    cfg = load_config("src/configs/brats_group_e.yaml")

    # config safety check (avoid KeyError crash)
    if not hasattr(cfg, "train"):
        raise KeyError("Config missing 'train' section")

    seeds = cfg.train.get("seed", [0])
    base_out = cfg.log["out_dir"]

    Path(base_out).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # multi-seed training
    # -------------------------
    for seed in seeds:

        set_seed(seed)

        run_dir = os.path.join(base_out, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n==============================")
        print(f"[RUN] seed={seed}")
        print(f"[DIR] {run_dir}")
        print(f"==============================\n")

        # -------------------------
        # data
        # -------------------------
        loaders = build_dataloaders(cfg.data)

        # -------------------------
        # model
        # -------------------------
        model = _build_model(cfg)

        # -------------------------
        # trainer
        # -------------------------
        trainer = MeanTeacherTrainer(model, cfg)

        # -------------------------
        # train
        # -------------------------
        trainer.fit(
            loaders=loaders,
            out_dir=run_dir
        )

        print(f"\n[DONE] seed={seed} saved to {run_dir}\n")


if __name__ == "__main__":
    main()