import torch
from monai.inferers import sliding_window_inference

from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.utils.metrics import compute_binary_metrics


def load_model(cfg, ckpt_path):

    model = HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"]
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=True)

    return model


@torch.no_grad()
def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    loaders = build_dataloaders(cfg.data)

    ckpt_path = cfg.log["out_dir"] + "/best.pt"

    model = load_model(cfg, ckpt_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics_sum = None
    n = 0

    roi = tuple(cfg.data.get("spatial_size", [96, 96, 96]))

    for batch in loaders["val"]:

        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = sliding_window_inference(
            x,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model
        )

        m = compute_binary_metrics(
            logits,
            y,
            threshold=cfg.inference.get("threshold", 0.45)
        )

        if metrics_sum is None:
            metrics_sum = {k: 0.0 for k in m.__dict__.keys()}

        for k in metrics_sum:
            metrics_sum[k] += getattr(m, k)

        n += 1

    for k in metrics_sum:
        metrics_sum[k] /= max(1, n)

    print("\n===== FINAL EVALUATION =====")
    for k, v in metrics_sum.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()