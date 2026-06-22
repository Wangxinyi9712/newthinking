from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianTeacher(nn.Module):
    """
    MC Dropout Teacher for uncertainty estimation
    """

    def __init__(self, model: nn.Module, mc_samples: int = 5):
        super().__init__()
        self.model = model
        self.mc_samples = mc_samples

    def forward(self, x):
        self.model.train()  # enable dropout

        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits, _ = self.model(x)
                preds.append(torch.sigmoid(logits))

        preds = torch.stack(preds, dim=0)  # [T, B, C, ...]
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)

        return mean, var