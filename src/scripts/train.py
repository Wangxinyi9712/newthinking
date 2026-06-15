from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config
from utils.seed import set_seed

warnings.filterwarnings(
    "ignore",
    message="single channel prediction, `include_background=False` ignored.",
)


def _to_plain_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]

    return obj


def _save_config_used(cfg, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(
        run_dir / "config_used.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _to_plain_dict(cfg),
            f,
            ensure_ascii=False,
            indent=2,
        )


def _build_model(cfg):
    dim = 3 if cfg.data.get("dim", "3d") == "3d" else 2

    return HybridUNet(
        in_channels=int(cfg.model["in_channels"]),
        out_channels=int(cfg.model["out_channels"]),
        channels=tuple(cfg.model["channels"]),
        dim=dim,
        use_transformer=bool(
            cfg.model.get("use_transformer", True)
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--run-name",
        default="",
        type=str,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        seed = int(args.seed)

    else:
        seeds = cfg.train.get("seed", [0])

        if isinstance(seeds, list):
            seed = int(seeds[0])
        else:
            seed = int(seeds)

    set_seed(seed)

    out_dir = Path(cfg.log["out_dir"])

    if args.run_name:
        out_dir = (
            out_dir.parent
            / f"{out_dir.name}_{args.run_name}"
        )

    run_dir = out_dir / f"seed_{seed}"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _save_config_used(cfg, run_dir)

    print(f"[Train] config={args.config}")
    print(f"[Train] seed={seed}")
    print(f"[Train] run_dir={run_dir}")

    # ===== 正确写法 =====
    loaders = build_dataloaders(cfg["data"])

    model = _build_model(cfg)

    trainer = MeanTeacherTrainer(
        model=model,
        cfg=cfg,
    )

    trainer.fit(
        loaders=loaders,
        save_dir=str(run_dir),
    )

    print("[Train] Done.")
    print(f"[Train] Best checkpoint : {run_dir/'best.pt'}")
    print(f"[Train] Last checkpoint : {run_dir/'last.pt'}")


if __name__ == "__main__":
    main()