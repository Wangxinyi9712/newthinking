import torch
import torch.nn as nn
import copy


class BayesianTeacher(nn.Module):

    def __init__(self, student):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.model.eval()

    def forward(self, x, T=5):

        preds = []

        for _ in range(T):
            self.model.train()
            with torch.no_grad():
                preds.append(torch.sigmoid(self.model(x)))

        preds = torch.stack(preds)

        return preds.mean(0), preds.var(0)