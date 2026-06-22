import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRefiner(nn.Module):
    """
    Lightweight refinement head (no real diffusion cost)
    """

    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, channels, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))