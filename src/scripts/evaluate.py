import torch

from src.utils.config import load_config
from src.models.seg_model import HybridUNet


# -------------------------
# model builder
# -------------------------
def build_model(cfg):

    model = HybridUNet(
        in_channels=cfg.model.get("in_channels", 4),
        out_channels=cfg.model.get("out_channels", 1),
        channels=cfg.model.get("channels", (16, 32, 64, 128)),
    )

    return model


# -------------------------
# checkpoint loader (FIXED)
# -------------------------
def load_model(cfg, ckpt_path):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = build_model(cfg)

    state = ckpt["student"]

    new_state = {}

    # -------------------------
    # key mapping (OLD → NEW)
    # -------------------------
    for k, v in state.items():

        k = k.replace("encoder", "enc1") \
             .replace("decoder", "dec1")  # fallback mapping

        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    return model.cuda().eval()


# -------------------------
# main
# -------------------------
def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    ckpt_path = cfg.eval["ckpt_path"]

    model = load_model(cfg, ckpt_path)

    print("[OK] model loaded successfully")


if __name__ == "__main__":
    main()