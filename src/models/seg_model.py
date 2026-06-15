import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    TMI unified interface:

    output:
        logits, features
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, return_features=False):

        out = self.model(x)

        # -----------------------------------------
        # CASE 1: model already returns tuple
        # -----------------------------------------
        if isinstance(out, (tuple, list)):
            logits = out[0]
            feat = out[1] if len(out) > 1 else logits
        else:
            logits = out
            feat = out

        if return_features:
            return logits, feat

        return logits