from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)

        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2,
             diffZ // 2, diffZ - diffZ // 2]
        )

        return self.conv(torch.cat([skip, x], dim=1))


class HybridUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base=32):
        super().__init__()

        self.inc = ConvBlock(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up3 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up1 = Up(base * 4, base * 2)

        self.outc = nn.Conv3d(base * 2, out_channels, 1)

    def forward(self, x, return_features=False):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        u3 = self.up3(b, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)

        logits = self.outc(u1)

        # ===============================
        # TMI FINAL: strict alignment
        # ===============================
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        if return_features:
            feat = torch.mean(b, dim=(2, 3, 4))
            return logits, feat

        return logits