import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class BayesianTeacher(nn.Module):
    """
    Uncertainty-aware teacher:
    - forward returns mean + variance (MC dropout style)
    """

    def __init__(self, model: nn.Module, mc_samples: int = 4):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.mc_samples = mc_samples

    @torch.no_grad()
    def forward(self, x):
        self.model.train()  # MC dropout style

        preds = []
        for _ in range(self.mc_samples):
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds.append(torch.sigmoid(logits))

        preds = torch.stack(preds, dim=0)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)

        return mean, var