import torch
from src.utils.config import load_config
from src.models.seg_model import HybridUNet


def build_model(cfg):
    model = HybridUNet(
        in_channels=cfg.model.get("in_channels", 4),
        out_channels=cfg.model.get("out_channels", 1),
        channels=cfg.model.get("channels", (32,64,128,256)),
    )
    return model


def load_model(cfg, ckpt_path):
    model = build_model(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("student", ckpt)

    # 🔥 critical: strict=False for compatibility across versions
    model.load_state_dict(state, strict=False)

    model.eval()
    return model


def main():
    cfg = load_config("src/configs/brats_group_e.yaml")

    ckpt_path = cfg.log.get("ckpt_path", "runs/best.pt")

    model = load_model(cfg, ckpt_path).cuda()

    print("[OK] model loaded for evaluation")


if __name__ == "__main__":
    main()