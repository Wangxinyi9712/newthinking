import torch
from src.models.seg_model import HybridUNet
from src.utils.config import load_config


def load_model(cfg, ckpt_path):

    model = HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"]
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["student"])

    return model


def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    ckpt_path = "runs/brats_group_e/seed_0/best.pt"

    model = load_model(cfg, ckpt_path)

    model.eval()


if __name__ == "__main__":
    main()