from __future__ import annotations

import torch
import torch.nn as nn

# =========================================================
# 1. IMPORT ORIGINAL MODEL (CRITICAL FIX)
# =========================================================
# 👉 这里必须指向你原来的实现文件
# 如果你的 HybridUNet 在别的文件，请按实际路径修改
from .hybrid_unet import HybridUNet


# =========================================================
# 2. TMI UNIFIED WRAPPER
# =========================================================
class ModelWrapper(nn.Module):
    """
    Standard TMI interface:

    return:
        logits
        or (logits, features)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, return_features: bool = False):

        out = self.model(x)

        # --------------------------
        # case: tuple output
        # --------------------------
        if isinstance(out, (tuple, list)):
            logits = out[0]
            feat = out[1] if len(out) > 1 else logits
        else:
            logits = out
            feat = out

        if return_features:
            return logits, feat

        return logits