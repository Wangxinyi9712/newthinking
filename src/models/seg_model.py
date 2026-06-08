from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    FrequencyEnhance,
    LiteTransformerEncoder,
    MultiScaleSkipAttention,
    PrototypeProjectionHead,
    UncertaintyGate,
)


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

    def forward(self, x: torch.Tensor):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dim: int):
        super().__init__()
        deconv = _conv_transpose(dim)
        self.up = deconv(in_ch, out_ch, 2, 2)
        self.skip_attn = MultiScaleSkipAttention(skip_ch, dim=dim)
        self.unc_gate = UncertaintyGate(skip_ch, dim=dim)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dim=dim)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        mode = "trilinear" if x.ndim == 5 else "bilinear"
        return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._resize_like(x, skip)
        skip = self.unc_gate(self.skip_attn(skip))
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
        conv = _conv(dim)

        self.enc1 = ConvBlock(in_channels, channels[0], dim)
        self.enc2 = ConvBlock(channels[0], channels[1], dim)
        self.enc3 = ConvBlock(channels[1], channels[2], dim)
        self.bottleneck = ConvBlock(channels[2], channels[3], dim)
        self.pool = pool(2)

        self.tr_path = LiteTransformerEncoder(channels[3]) if use_transformer else nn.Identity()
        self.cnn_path = ConvBlock(channels[3], channels[3], dim)
        self.fuse = nn.Sequential(conv(channels[3] * 2, channels[3], 1), nn.GELU(), conv(channels[3], channels[3], 1))
        self.freq_enhance = FrequencyEnhance(channels[3], dim=dim, alpha=0.12)

        self.up3 = UpBlock(channels[3], channels[2], channels[2], dim)
        self.up2 = UpBlock(channels[2], channels[1], channels[1], dim)
        self.up1 = UpBlock(channels[1], channels[0], channels[0], dim)

        self.out = conv(channels[0], out_channels, 1)
        self.proto_head = PrototypeProjectionHead(channels[3], proj_ch=min(64, channels[3]), dim=dim)

    def _merge_features(self, b: torch.Tensor) -> torch.Tensor:
        b = self.fuse(torch.cat([self.tr_path(b), self.cnn_path(b)], dim=1))
        b = self.freq_enhance(b)
        return b

    def forward(self, x: torch.Tensor, return_features: bool = False):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self._merge_features(self.bottleneck(self.pool(s3)))

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        logits = self.out(d1)

        if return_features:
            proto_feat = self.proto_head(b)
            if proto_feat.shape[2:] != b.shape[2:]:
                mode = "trilinear" if b.ndim == 5 else "bilinear"
                proto_feat = F.interpolate(proto_feat, size=b.shape[2:], mode=mode, align_corners=False)
            return logits, torch.cat([b, proto_feat], dim=1)
        return logits