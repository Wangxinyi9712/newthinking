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

    # ❗ strict=False to survive legacy mismatch
    model.load_state_dict(state, strict=False)

    model.eval()
    return model