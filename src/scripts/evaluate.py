import os
import torch
from monai.inferers import sliding_window_inference

from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.utils.metrics import compute_binary_metrics


def find_ckpt(base_dir):

    candidates = [
        os.path.join(base_dir, "best.pt"),
        os.path.join(base_dir, "last.pt"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"No checkpoint found in {base_dir}. "
        f"Expected best.pt or last.pt"
    )


def load_model(cfg, ckpt_path):

    model = HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"],
        channels=cfg.model.get("channels", [32, 64, 128, 256]),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=True)

    return model


@torch.no_grad()
def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    loaders = build_dataloaders(cfg.data)

    base_dir = cfg.log["out_dir"]

    ckpt_path = find_ckpt(base_dir)

    print(f"[CKPT] using: {ckpt_path}")

    model = load_model(cfg, ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    roi = tuple(cfg.data.get("spatial_size", [96, 96, 96]))

    metric_sum = None
    n = 0

    for batch in loaders["val"]:

        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = sliding_window_inference(
            x,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model
        )

        m = compute_binary_metrics(logits, y)

        if metric_sum is None:
            metric_sum = {k: 0.0 for k in m.__dict__.keys()}

        for k in metric_sum:
            metric_sum[k] += float(getattr(m, k))

        n += 1

    for k in metric_sum:
        metric_sum[k] /= max(1, n)

    print("\n===== FINAL RESULT =====")
    for k, v in metric_sum.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()