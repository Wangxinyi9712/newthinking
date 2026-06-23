import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRefiner(nn.Module):

    def __init__(self, channels=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, channels, 1),
        )

    def forward(self, x):

        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise

        residual = self.net(x_noisy)

        return torch.sigmoid(x + residual)