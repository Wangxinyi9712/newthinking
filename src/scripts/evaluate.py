from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.utils.config import load_config
from src.utils.metrics import compute_binary_metrics


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


def resolve_ckpt(cfg, ckpt_arg: str, seed: int) -> Path:
    if ckpt_arg in {"best", "last"}:
        return Path(cfg.log["out_dir"]) / f"seed_{seed}" / f"{ckpt_arg}.pt"

    p = Path(ckpt_arg)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    return p


def load_model(
    cfg,
    ckpt_path: Path,
    device: torch.device,
    allow_partial: bool = False,
    source: str = "teacher",
) -> HybridUNet:
    model = build_model(cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if source == "teacher":
        state = ckpt.get("teacher", ckpt.get("student", ckpt))
    elif source == "student":
        state = ckpt.get("student", ckpt.get("teacher", ckpt))
    else:
        raise ValueError(f"Unsupported source: {source}. Use teacher or student.")

    try:
        incompatible = model.load_state_dict(state, strict=not allow_partial)
    except RuntimeError as e:
        raise RuntimeError(
            "\n[Checkpoint mismatch]\n"
            f"Checkpoint cannot be strictly loaded: {ckpt_path}\n"
            "This usually means the checkpoint was trained with an older model definition.\n"
            "For valid experiments, retrain after replacing the code.\n"
            "For debugging only, pass --allow-partial.\n"
        ) from e

    if allow_partial:
        missing, unexpected = incompatible

        if missing:
            print(f"[WARN] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "src" / "configs" / "brats_group_e.yaml"))
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = load_config(config_path)
    normalize_config_paths(cfg)

    if args.max_val_batches > 0:
        cfg.train["max_val_batches"] = int(args.max_val_batches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt(cfg, args.ckpt, seed=int(args.seed))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        allow_partial=bool(args.allow_partial),
        source=args.source,
    )

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
    max_val_batches = int(cfg.train.get("max_val_batches", 0) or 0)

    for batch in loaders["val"]:
        x = batch["image"].to(device).float()
        y = batch["label"].to(device).float()

        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        metrics = compute_binary_metrics(logits.float(), y.float(), threshold=threshold)

        totals["dice"] += float(metrics.dice)
        totals["iou"] += float(metrics.iou)
        totals["precision"] += float(metrics.precision)
        totals["recall"] += float(metrics.recall)
        totals["f1"] += float(metrics.f1)
        totals["minority_f1"] += float(metrics.minority_f1)
        totals["hd95"] += float(metrics.hd95)

        n += 1

        if max_val_batches > 0 and n >= max_val_batches:
            break

    n = max(1, n)
    results = {k: v / n for k, v in totals.items()}

    results["config"] = str(config_path)
    results["ckpt"] = str(ckpt_path)
    results["seed"] = int(args.seed)
    results["source"] = args.source

    print("===== Evaluation Results =====")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = ckpt_path.parent / f"eval_results_{args.source}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()