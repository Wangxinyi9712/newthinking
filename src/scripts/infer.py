from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.utils.config import load_config
from src.utils.seed import set_seed


DEFAULT_CONFIG = PROJECT_ROOT / "src" / "configs" / "ours.yaml"


def normalize_config_paths(cfg) -> None:
    split_file = cfg.data.get("split_file", cfg.data.get("split_json"))

    if split_file is not None:
        p = Path(split_file)
        if not p.is_absolute():
            cfg.data["split_file"] = str(PROJECT_ROOT / p)

    out_dir = cfg.log.get("out_dir", "runs/ours")
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


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def resolve_ckpt(cfg, ckpt_arg: str, seed: int) -> Path:
    if ckpt_arg in {"best", "last"}:
        return Path(cfg.log["out_dir"]) / f"seed_{seed}" / f"{ckpt_arg}.pt"

    p = Path(ckpt_arg)

    if not p.is_absolute():
        p = PROJECT_ROOT / p

    return p


def load_state_into_model(
    cfg,
    ckpt_path: Path,
    device: torch.device,
    source: str,
) -> HybridUNet:
    model = build_model(cfg).to(device)
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    if source == "teacher":
        state = ckpt.get("teacher", ckpt.get("student", ckpt))
    elif source == "student":
        state = ckpt.get("student", ckpt.get("teacher", ckpt))
    else:
        raise ValueError(f"Unsupported source for single model: {source}")

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


@torch.no_grad()
def save_case(out_dir: Path, idx: int, image: torch.Tensor, prob: torch.Tensor, pred: torch.Tensor) -> None:
    case_dir = out_dir / f"case_{idx:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    torch.save(image.detach().cpu(), case_dir / "image.pt")
    torch.save(prob.detach().cpu(), case_dir / "prob.pt")
    torch.save(pred.detach().cpu(), case_dir / "pred.pt")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config file. Default is the revised Our Method config.",
    )
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="teacher", choices=["teacher", "student", "ensemble"])
    parser.add_argument("--out", type=str, default="runs/ours/infer")
    parser.add_argument("--max-cases", default=-1, type=int)

    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    normalize_config_paths(cfg)

    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_ckpt(cfg, args.ckpt, seed=int(args.seed))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.source == "ensemble":
        student = load_state_into_model(cfg, ckpt_path, device, source="student")
        teacher = load_state_into_model(cfg, ckpt_path, device, source="teacher")
        model = None
    else:
        model = load_state_into_model(cfg, ckpt_path, device, source=args.source)
        student = None
        teacher = None

    loaders = build_dataloaders(cfg.data)

    out_dir = Path(args.out)

    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(cfg.inference.get("threshold", 0.5))
    saved = 0

    for i, batch in enumerate(loaders["val"]):
        if args.max_cases > 0 and i >= args.max_cases:
            break

        x = batch["image"].to(device).float()

        if args.source == "ensemble":
            logits_s = student(x)
            logits_t = teacher(x)

            if isinstance(logits_s, tuple):
                logits_s = logits_s[0]

            if isinstance(logits_t, tuple):
                logits_t = logits_t[0]

            logits = 0.5 * (logits_s + logits_t)
        else:
            logits = model(x)

            if isinstance(logits, tuple):
                logits = logits[0]

        prob = torch.sigmoid(logits.float())
        pred = (prob > threshold).float()

        save_case(out_dir, i, x, prob, pred)
        saved += 1

    print(
        {
            "saved_cases": saved,
            "threshold": threshold,
            "ckpt": str(ckpt_path),
            "source": args.source,
            "output_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()