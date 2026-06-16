from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim):
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _norm(dim):
    return nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super().__init__()
        conv = _conv(dim)
        norm = _norm(dim)

        self.block = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            conv(out_ch, out_ch, 3, padding=1),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HybridUNet(nn.Module):
    """
    TMI-grade simplified UNet baseline (stable + debug-friendly)
    输出 logits: (B, 1, H, W, D)
    """

    def __init__(self, in_channels=4, out_channels=1, base_channels=32, dim=3):
        super().__init__()
        self.dim = dim

        self.enc1 = ConvBlock(in_channels, base_channels, dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dim)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dim)

        conv = _conv(dim)

        self.pool = nn.MaxPool3d(2) if dim == 3 else nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, dim)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2) if dim == 3 else nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, dim)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2) if dim == 3 else nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dim)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2) if dim == 3 else nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dim)

        self.out_conv = conv(base_channels, out_channels, 1)

    def forward(self, x, return_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out_conv(d1)

        if return_features:
            return logits, {"e1": e1, "e2": e2, "e3": e3}
        return logits, None