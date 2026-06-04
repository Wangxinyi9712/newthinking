from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.datasets import build_dataloaders
from engine.infer import run_inference
from models.seg_model import HybridUNet
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", choices=["teacher", "student", "ensemble"], default="teacher")
    args = parser.parse_args()

    cfg_obj = load_config(args.config)
    cfg = {k: getattr(cfg_obj, k) for k in ["data", "model", "train", "loss", "inference", "log"]}

    model = HybridUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=tuple(cfg["model"].get("channels", [32, 64, 128, 256])),
        dim=3 if cfg["data"].get("dim", "3d") == "3d" else 2,
        use_transformer=cfg["model"].get("use_transformer", True),
    )

    state = torch.load(args.ckpt, map_location="cpu")
    if args.source == "ensemble":
        model.load_state_dict(state["teacher"], strict=True)
    else:
        model.load_state_dict(state[args.source], strict=True)

    loaders = build_dataloaders(cfg["data"])
    run_inference(model, loaders["val"], args.out)


if __name__ == "__main__":
    main()