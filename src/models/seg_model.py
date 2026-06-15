from __future__ import annotations

import torch.nn as nn
from .hybrid_unet import HybridUNet


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, return_features=False):
        out = self.model(x)

        if isinstance(out, (tuple, list)):
            logits, feat = out[0], out[1]
        else:
            logits, feat = out, out

        return (logits, feat) if return_features else logits