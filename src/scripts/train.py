from __future__ import annotations

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
    cfg = load_config("src/configs/brats_group_e.yaml")

    seeds = cfg.train.get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    loaders = build_dataloaders(cfg.data)

    for seed in seeds:
        seed = int(seed)
        set_seed(seed)

        model = build_model(cfg)
        trainer = MeanTeacherTrainer(model, cfg)

        run_dir = Path(cfg.log["out_dir"]) / f"seed_{seed}"
        trainer.fit(loaders, str(run_dir))

        print(f"[DONE] seed={seed}, outputs saved to {run_dir}")


if __name__ == "__main__":
    main()