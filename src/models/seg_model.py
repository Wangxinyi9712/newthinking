from __future__ import annotations

import torch
import torch.nn as nn

from .modules import LiteTransformerEncoder, MultiScaleSkipAttention


def _conv(dim: int):
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _conv_transpose(dim: int):
    return nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d


def _norm(dim: int):
    return nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d


def _maxpool(dim: int):
    return nn.MaxPool3d if dim == 3 else nn.MaxPool2d


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int):
        super().__init__()
        conv = _conv(dim)
        norm = _norm(dim)
        self.net = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1),
            norm(out_ch),
            nn.LeakyReLU(inplace=True),
            conv(out_ch, out_ch, 3, padding=1),
            norm(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dim: int):
        super().__init__()
        deconv = _conv_transpose(dim)
        self.up = deconv(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = MultiScaleSkipAttention(skip_ch, dim=dim)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dim=dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attn(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HybridUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        dim: int = 3,
        use_transformer: bool = True,
    ):
        super().__init__()
        pool = _maxpool(dim)
        self.enc1 = ConvBlock(in_channels, channels[0], dim)
        self.enc2 = ConvBlock(channels[0], channels[1], dim)
        self.enc3 = ConvBlock(channels[1], channels[2], dim)
        self.bottleneck = ConvBlock(channels[2], channels[3], dim)
        self.pool = pool(2)

        self.transformer = LiteTransformerEncoder(channels[3]) if use_transformer else nn.Identity()

        self.up3 = UpBlock(channels[3], channels[2], channels[2], dim)
        self.up2 = UpBlock(channels[2], channels[1], channels[1], dim)
        self.up1 = UpBlock(channels[1], channels[0], channels[0], dim)

        conv = _conv(dim)
        self.out = conv(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))
        b = self.transformer(b)

        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        logits = self.out(x)
        if return_features:
            return logits, b
        return logits