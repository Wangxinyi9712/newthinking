import torch
from src.models.seg_model import HybridUNet
from src.utils.config import load_config


def load_model(cfg, ckpt_path):

    model = HybridUNet(
        in_channels=cfg.model.get("in_channels", 4),
        out_channels=cfg.model.get("out_channels", 1),
        channels=cfg.model.get("channels", (32,64,128,256))
    ).cuda()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("student", ckpt)

    model.load_state_dict(state, strict=True)

    model.eval()
    return model


def main():
    cfg = load_config("src/configs/brats_group_e.yaml")

    ckpt_path = cfg.log.get("ckpt", "runs/seed_0/best.pt")

    model = load_model(cfg, ckpt_path)

    print("[OK] model loaded")

if __name__ == "__main__":
    main()