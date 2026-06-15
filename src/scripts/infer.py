from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config
from utils.seed import set_seed


def _resolve_ckpt(
    ckpt_arg: str,
    config_path: str,
    split: str = "seed_0",
):
    p = Path(ckpt_arg)

    if p.suffix == ".pt":
        return p

    alias = ckpt_arg.lower()

    if alias not in {"best", "last"}:
        raise ValueError(
            "ckpt must be path or best/last"
        )

    cfg = load_config(config_path)

    return (
        Path(cfg.log["out_dir"])
        / split
        / f"{alias}.pt"
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


@torch.no_grad()
def _save_case(
    out_dir,
    idx,
    image,
    prob,
    pred,
):
    case_dir = out_dir / f"case_{idx:04d}"

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        image.cpu(),
        case_dir / "image.pt",
    )

    torch.save(
        prob.cpu(),
        case_dir / "prob.pt",
    )

    torch.save(
        pred.cpu(),
        case_dir / "pred.pt",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        default="best",
    )

    parser.add_argument(
        "--source",
        default="teacher",
        choices=[
            "teacher",
            "student",
            "ensemble",
        ],
    )

    parser.add_argument(
        "--split",
        default="seed_0",
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--max-cases",
        default=-1,
        type=int,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(args.seed)

    ckpt_path = _resolve_ckpt(
        args.ckpt,
        args.config,
        args.split,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = _build_model(cfg)

    ckpt = torch.load(
        str(ckpt_path),
        map_location="cpu",
    )

    if "student" in ckpt:
        model.load_state_dict(
            ckpt["student"],
            strict=False,
        )
    else:
        model.load_state_dict(
            ckpt,
            strict=False,
        )

    # ===== 正确写法 =====
    loaders = build_dataloaders(cfg["data"])

    trainer = MeanTeacherTrainer(
        model,
        cfg,
    )

    if "teacher" in ckpt:
        trainer.teacher.load_state_dict(
            ckpt["teacher"],
            strict=False,
        )

    if "student" in ckpt:
        trainer.student.load_state_dict(
            ckpt["student"],
            strict=False,
        )

    trainer.student.eval()
    trainer.teacher.eval()

    out_dir = Path(args.out)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold = float(
        cfg.inference.get(
            "threshold",
            0.45,
        )
    )

    device = trainer.device

    saved = 0

    for i, batch in enumerate(loaders["val"]):

        if (
            args.max_cases > 0
            and i >= args.max_cases
        ):
            break

        x = batch["image"].to(device)

        if args.source == "ensemble":

            z1, _ = trainer.student(x)
            z2, _ = trainer.teacher(x)

            logits = (z1 + z2) * 0.5

        elif args.source == "teacher":

            logits, _ = trainer.teacher(x)

        else:

            logits, _ = trainer.student(x)

        prob = torch.sigmoid(
            logits.float()
        )

        pred = (
            prob > threshold
        ).float()

        _save_case(
            out_dir,
            i,
            x,
            prob,
            pred,
        )

        saved += 1

    print(
        {
            "saved_cases": saved,
            "threshold": threshold,
            "ckpt": str(ckpt_path),
            "output_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()