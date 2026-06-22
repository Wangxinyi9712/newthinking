import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRefiner(nn.Module):
    """
    lightweight diffusion-inspired refinement:
    - iterative denoising with residual conv
    """

    def __init__(self, channels=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(channels, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, channels, 3, padding=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 4 else x
        noise = self.net(x)
        return torch.sigmoid(x - noise)