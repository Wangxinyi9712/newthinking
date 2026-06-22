import torch
import torch.nn as nn
from .modules import *


class HybridUNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, channels=[32, 64, 128, 256]):
        super().__init__()

        self.enc1 = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        self.enc2 = nn.Conv3d(channels[0], channels[1], 3, padding=1)
        self.enc3 = nn.Conv3d(channels[1], channels[2], 3, padding=1)

        self.decoder = nn.Conv3d(channels[2], out_channels, 1)

        self.feature_proj = nn.Conv3d(channels[2], 64, 1)

    def forward(self, x, return_features=False):

        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(x1))
        feat = torch.relu(self.enc3(x2))

        out = self.decoder(feat)

        if return_features:
            return out, self.feature_proj(feat)

        return out