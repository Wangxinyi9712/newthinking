from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.utils.config import load_config
from src.utils.metrics import compute_binary_metrics


def build_model(cfg) -> HybridUNet:
    return HybridUNet(
        in_channels=int(cfg.model.get("in_channels", 4)),
        out_channels=int(cfg.model.get("out_channels", 1)),
        channels=tuple(cfg.model.get("channels", [16, 32, 64, 128])),
        use_transformer=bool(cfg.model.get("use_transformer", False)),
    )


def resolve_ckpt(cfg, ckpt_arg: str) -> Path:
    if ckpt_arg in {"best", "last"}:
        return Path(cfg.log["out_dir"]) / "seed_0" / f"{ckpt_arg}.pt"

    return Path(ckpt_arg)


def load_model(cfg, ckpt_path: Path, device: torch.device) -> HybridUNet:
    model = build_model(cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("teacher", ckpt.get("student", ckpt))

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brats_group_e.yaml")
    parser.add_argument("--ckpt", type=str, default="best")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt(cfg, args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(cfg, ckpt_path, device)
    loaders = build_dataloaders(cfg.data)

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "minority_f1": 0.0,
        "hd95": 0.0,
    }

    n = 0
    threshold = float(cfg.inference.get("threshold", 0.5))

    for batch in loaders["val"]:
        x = batch["image"].to(device).float()
        y = batch["label"].to(device).float()

        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        m = compute_binary_metrics(logits.float(), y.float(), threshold=threshold)

        totals["dice"] += float(m.dice)
        totals["iou"] += float(m.iou)
        totals["precision"] += float(m.precision)
        totals["recall"] += float(m.recall)
        totals["f1"] += float(m.f1)
        totals["minority_f1"] += float(m.minority_f1)
        totals["hd95"] += float(m.hd95)
        n += 1

    n = max(1, n)

    print("===== Evaluation Results =====")
    for k, v in totals.items():
        print(f"{k}: {v / n:.6f}")


if __name__ == "__main__":
    main()