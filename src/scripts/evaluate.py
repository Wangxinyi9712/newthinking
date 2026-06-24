import torch
from src.utils.config import load_config
from src.models.seg_model import HybridUNet


def load_model(cfg, ckpt_path):

    model = HybridUNet(
        in_channels=cfg.model["in_channels"],
        out_channels=cfg.model["out_channels"],
        channels=cfg.model["channels"]
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("student", ckpt)

    model.load_state_dict(state, strict=False)
    model.eval()

    return model


def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    ckpt_path = cfg.log["ckpt_path"]

    model = load_model(cfg, ckpt_path).cuda()

    print("[OK] Camera-ready evaluation loaded")


if __name__ == "__main__":
    main()