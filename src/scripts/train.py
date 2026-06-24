from __future__ import annotations

import shutil
from pathlib import Path

from src.data.datasets import build_dataloaders
from src.engine.trainer import MeanTeacherTrainer
from src.models.seg_model import HybridUNet
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_model(cfg) -> HybridUNet:
    return HybridUNet(
        in_channels=int(cfg.model.get("in_channels", 4)),
        out_channels=int(cfg.model.get("out_channels", 1)),
        channels=tuple(cfg.model.get("channels", [16, 32, 64, 128])),
        use_transformer=bool(cfg.model.get("use_transformer", False)),
    )


def main() -> None:
    config_path = Path("src/configs/brats_group_e.yaml")
    cfg = load_config(config_path)

    seeds = cfg.train.get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    for seed in seeds:
        seed = int(seed)

        # 关键：必须先 set_seed，再构建包含随机 crop/augment 的 dataloader
        set_seed(seed)

        run_dir = Path(cfg.log["out_dir"]) / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 保存本次实验配置，保证可复现
        shutil.copy2(config_path, run_dir / "config.yaml")

        loaders = build_dataloaders(cfg.data)

        model = build_model(cfg)
        trainer = MeanTeacherTrainer(model, cfg)

        trainer.fit(loaders, str(run_dir))

        print(f"[DONE] seed={seed}, outputs saved to {run_dir}")


if __name__ == "__main__":
    main()