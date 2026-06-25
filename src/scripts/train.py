from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.datasets import build_dataloaders
from src.engine.trainer import MeanTeacherTrainer
from src.models.seg_model import HybridUNet
from src.utils.config import load_config
from src.utils.seed import set_seed


def normalize_config_paths(cfg) -> None:
    split_file = cfg.data.get("split_file", cfg.data.get("split_json"))
    if split_file is not None:
        p = Path(split_file)
        if not p.is_absolute():
            cfg.data["split_file"] = str(PROJECT_ROOT / p)

    out_dir = cfg.log.get("out_dir", "runs/default")
    p = Path(out_dir)
    if not p.is_absolute():
        cfg.log["out_dir"] = str(PROJECT_ROOT / p)


def build_model(cfg) -> HybridUNet:
    return HybridUNet(
        in_channels=int(cfg.model.get("in_channels", 4)),
        out_channels=int(cfg.model.get("out_channels", 1)),
        channels=tuple(cfg.model.get("channels", [16, 32, 64, 128])),
        use_transformer=bool(cfg.model.get("use_transformer", False)),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "src" / "configs" / "brats_group_e.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = load_config(config_path)
    normalize_config_paths(cfg)

    if args.epochs is not None:
        cfg.train["epochs"] = int(args.epochs)

    if args.smoke:
        cfg.train["epochs"] = min(int(cfg.train.get("epochs", 2)), 2)
        cfg.train["max_train_steps"] = int(cfg.train.get("max_train_steps", 5) or 5)
        cfg.train["max_val_batches"] = int(cfg.train.get("max_val_batches", 5) or 5)
        cfg.train["log_every"] = 1
        cfg.loss["lambda_spec"] = 0.0

    seeds = cfg.train.get("seed", [0])
    if args.seed is not None:
        seeds = [args.seed]

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