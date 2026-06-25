from __future__ import annotations

import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Make this script runnable in both ways:
#   python src/scripts/train.py
#   python -m src.scripts.train
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    config_path = PROJECT_ROOT / "src" / "configs" / "brats_group_e.yaml"
    cfg = load_config(config_path)

    seeds = cfg.train.get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    for seed in seeds:
        seed = int(seed)
        set_seed(seed)

        run_dir = Path(cfg.log["out_dir"]) / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(config_path, run_dir / "config.yaml")

        loaders = build_dataloaders(cfg.data)

        model = build_model(cfg)
        trainer = MeanTeacherTrainer(model, cfg)

        trainer.fit(loaders, str(run_dir))

        print(f"[DONE] seed={seed}, outputs saved to {run_dir}")


if __name__ == "__main__":
    main()