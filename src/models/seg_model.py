import torch
import torch.nn as nn


class HybridUNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, 1)
        )

    def forward(self, x, return_features=False):

        f = self.encoder(x)
        out = self.decoder(f)

        if return_features:
            return out, f

        return out