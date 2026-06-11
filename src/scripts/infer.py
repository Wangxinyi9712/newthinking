from __future__ import annotations

import argparse
from pathlib import Path

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
    out_dir = Path(cfg["log"]["out_dir"]) / split
    return out_dir / f"{alias}.pt"


@torch.no_grad()
def _save_case(out_dir: Path, idx: int, image: torch.Tensor, prob: torch.Tensor, pred: torch.Tensor):
    """
    保存为pt，避免引入额外依赖（nibabel等）。
    如需nii.gz可后续扩展。
    """
    case_dir = out_dir / f"case_{idx:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    torch.save(image.cpu(), case_dir / "image.pt")
    torch.save(prob.cpu(), case_dir / "prob.pt")
    torch.save(pred.cpu(), case_dir / "pred.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        help="Checkpoint path (*.pt) OR alias {best,last}. Default: best",
    )
    parser.add_argument("--source", type=str, default="teacher", choices=["teacher", "student", "ensemble"])
    parser.add_argument("--split", type=str, default="seed_0", help="When ckpt is alias, use this run split dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="output dir")
    parser.add_argument("--max-cases", type=int, default=-1, help="limit number of inference cases")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    ckpt_path = _resolve_ckpt(args.ckpt, args.config, split=args.split)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dim = 3 if cfg["data"].get("dim", "3d") == "3d" else 2
    model = HybridUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        channels=tuple(cfg["model"]["channels"]),
        dim=dim,
        use_transformer=bool(cfg["model"].get("use_transformer", True)),
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "student" in ckpt:
        model.load_state_dict(ckpt["student"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    loaders = build_dataloaders(cfg)
    trainer = MeanTeacherTrainer(model, cfg)

    if "teacher" in ckpt:
        trainer.teacher.load_state_dict(ckpt["teacher"], strict=False)
    if "student" in ckpt:
        trainer.student.load_state_dict(ckpt["student"], strict=False)

    trainer.student.eval()
    trainer.teacher.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(cfg["inference"].get("threshold", 0.40))
    device = trainer.device

    n = 0
    for i, batch in enumerate(loaders["val"]):
        if args.max_cases > 0 and i >= args.max_cases:
            break

        x = batch["image"].to(device)

        with (torch.amp.autocast("cuda") if trainer.use_amp else torch.no_grad()):
            if args.source == "ensemble":
                z1, _ = trainer.student(x)
                z2, _ = trainer.teacher(x)
                logits = 0.5 * z1 + 0.5 * z2
            elif args.source == "teacher":
                logits, _ = trainer.teacher(x)
            else:
                logits, _ = trainer.student(x)

        prob = torch.sigmoid(logits.float())
        pred = (prob > threshold).float()

        _save_case(out_dir, i, x, prob, pred)
        n += 1

    summary = {
        "ckpt": str(ckpt_path),
        "source": args.source,
        "saved_cases": n,
        "output_dir": str(out_dir),
        "threshold": threshold,
    }
    print(summary)


if __name__ == "__main__":
    main()