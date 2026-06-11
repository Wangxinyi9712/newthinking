from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# ------------------------------
# 确保可以导入 data 和 engine 模块
ROOT = Path(__file__).resolve().parents[1]  # 指向 src
sys.path.insert(0, str(ROOT))
# ------------------------------

import torch

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config
from utils.seed import set_seed


def _resolve_ckpt(ckpt_arg: str, config_path: str, split: str = "seed_0") -> Path:
    """
    支持三种传法：
    1) --ckpt /abs/or/rel/path/to/xxx.pt
    2) --ckpt best
    3) --ckpt last
    """
    p = Path(ckpt_arg)
    if p.suffix == ".pt":
        return p

    alias = ckpt_arg.lower().strip()
    if alias not in {"best", "last"}:
        raise ValueError(f"Unsupported --ckpt value: {ckpt_arg}. Use path/*.pt or alias best|last.")

    cfg = load_config(config_path)
    out_dir = Path(cfg.log["out_dir"]) / split
    return out_dir / f"{alias}.pt"


def _load_model_from_ckpt(cfg: dict, ckpt_path: Path, source: str):
    dim = 3 if cfg.data.get("dim", "3d") == "3d" else 2

    model = HybridUNet(
        in_channels=int(cfg.model["in_channels"]),
        out_channels=int(cfg.model["out_channels"]),
        channels=tuple(cfg.model["channels"]),
        dim=dim,
        use_transformer=bool(cfg.model.get("use_transformer", True)),
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # 优先按 source 加载；若不存在则回退
    if source == "student":
        key = "student"
    elif source == "teacher":
        key = "teacher"
    elif source == "ensemble":
        key = "student"  # ensemble会在trainer.evaluate里用teacher+student
    else:
        raise ValueError(f"Invalid source: {source}")

    if key in ckpt:
        model.load_state_dict(ckpt[key], strict=False)
    else:
        # 兼容仅保存裸state_dict的情况
        model.load_state_dict(ckpt, strict=False)

    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        help="Checkpoint path (*.pt) OR alias {best,last}. Default: best",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="teacher",
        choices=["teacher", "student", "ensemble"],
        help="Which source to evaluate",
    )
    parser.add_argument("--split", type=str, default="seed_0", help="When ckpt is alias, use this run split dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", type=str, default="", help="optional output json path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    ckpt_path = _resolve_ckpt(args.ckpt, args.config, split=args.split)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, _ = _load_model_from_ckpt(cfg, ckpt_path, source=args.source)

    loaders = build_dataloaders(cfg.data)
    trainer = MeanTeacherTrainer(model, cfg)

    # 若source是teacher，尝试恢复teacher权重
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "teacher" in ckpt:
        trainer.teacher.load_state_dict(ckpt["teacher"], strict=False)
    if "student" in ckpt:
        trainer.student.load_state_dict(ckpt["student"], strict=False)

    metrics = trainer.evaluate(
        loaders["val"],
        use_teacher_ema=(args.source == "teacher"),
        use_ensemble=(args.source == "ensemble"),
    )

    result = {
        "ckpt": str(ckpt_path),
        "source": args.source,
        "dice": float(metrics.dice),
        "iou": float(metrics.iou),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "minority_f1": float(metrics.minority_f1),
        "hd95": float(metrics.hd95),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
