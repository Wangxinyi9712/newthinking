from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config
from utils.seed import set_seed


warnings.filterwarnings(
    "ignore",
    message="single channel prediction, `include_background=False` ignored.",
)


def _to_plain_dict(cfg):
    """将可能的 OmegaConf/其他配置对象转为原生 dict，便于 json dump。"""
    if isinstance(cfg, dict):
        return {k: _to_plain_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_to_plain_dict(v) for v in cfg]
    return cfg


def _build_model(cfg: dict) -> HybridUNet:
    dim = 3 if cfg["data"].get("dim", "3d") == "3d" else 2
    model = HybridUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        channels=tuple(cfg["model"]["channels"]),
        dim=dim,
        use_transformer=bool(cfg["model"].get("use_transformer", True)),
    )
    return model


def _save_config_used(cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(_to_plain_dict(cfg), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    parser.add_argument("--run-name", type=str, default="", help="Optional suffix for output dir")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 读取seed策略：优先CLI，其次config中的第一个seed，最后0
    if args.seed is not None:
        seed = int(args.seed)
    else:
        seeds = cfg.get("train", {}).get("seed", [0])
        seed = int(seeds[0] if isinstance(seeds, list) and len(seeds) > 0 else 0)

    set_seed(seed)

    # 输出目录：log.out_dir/seed_{seed}
    base_out = Path(cfg["log"]["out_dir"])
    if args.run_name:
        base_out = base_out.parent / f"{base_out.name}_{args.run_name}"
    run_dir = base_out / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存实际配置
    _save_config_used(cfg, run_dir)

    # 构建数据
    loaders = build_dataloaders(cfg["data"])

    # 构建模型与trainer
    model = _build_model(cfg)
    trainer = MeanTeacherTrainer(model, cfg)

    print(f"[Train] config={args.config}")
    print(f"[Train] seed={seed}")
    print(f"[Train] run_dir={run_dir}")

    # 开始训练（trainer内部会保存best.pt与last.pt）
    trainer.fit(loaders, str(run_dir))

    print("[Train] Done.")
    print(f"[Train] Checkpoints: {run_dir / 'best.pt'} , {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()