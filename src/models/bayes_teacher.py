import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianTeacher(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x, T=5):

        preds = []

        for _ in range(T):
            self.model.train()  # dropout enable
            with torch.no_grad():
                out = torch.sigmoid(self.model(x))
                preds.append(out)

        preds = torch.stack(preds)

        mean = preds.mean(0)
        var = preds.var(0)

        return mean, var