from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic 3D convolution block:
        Conv3d -> InstanceNorm3d -> LeakyReLU
        Conv3d -> InstanceNorm3d -> LeakyReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerBottleneck(nn.Module):
    """
    Lightweight transformer bottleneck for 3D feature maps.

    Input:
        x: [B, C, D, H, W]

    The bottleneck feature resolution is usually small, e.g. 12x12x12 for 96^3 input,
    so flattening spatial tokens is acceptable.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        num_heads = max(1, min(num_heads, channels))
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        out = tokens.transpose(1, 2).reshape(b, c, d, h, w)
        return out


class HybridUNet(nn.Module):
    """
    3D U-Net backbone used by all experiments.

    Supports:
      - segmentation logits
      - feature map for prototype contrastive learning
      - SDF head for boundary-aware training

    Default behavior is backward compatible:
        model(x) -> logits

    Feature mode:
        model(x, return_features=True) -> logits, feat

    SDF mode:
        model(x, return_sdf=True) -> logits, sdf

    Feature + SDF mode:
        model(x, return_features=True, return_sdf=True) -> logits, feat, sdf
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: Sequence[int] = (16, 32, 64, 128),
        use_transformer: bool = False,
    ):
        super().__init__()

        if len(channels) != 4:
            raise ValueError(f"channels must contain exactly 4 values, got {channels}")

        c1, c2, c3, c4 = [int(v) for v in channels]

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = (c1, c2, c3, c4)
        self.use_transformer = bool(use_transformer)

        self.enc1 = ConvBlock(self.in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        if self.use_transformer:
            self.bottleneck = TransformerBottleneck(c4, num_heads=4, depth=1)
        else:
            self.bottleneck = nn.Identity()

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.seg_head = nn.Conv3d(c1, self.out_channels, kernel_size=1)

        feature_mid = min(32, c1)
        self.feature_head = nn.Sequential(
            nn.Conv3d(c1, feature_mid, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_mid, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(feature_mid, 32, kernel_size=1),
        )

        self.sdf_head = nn.Sequential(
            nn.Conv3d(c1, feature_mid, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_mid, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(feature_mid, 1, kernel_size=1),
        )

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Align upsampled tensor to skip-connection tensor.
        This avoids size mismatch caused by 3D pooling / transpose convolution.
        """
        if x.shape[2:] == ref.shape[2:]:
            return x

        return F.interpolate(
            x,
            size=ref.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_feat: bool = False,
        return_sdf: bool = False,
    ):
        return_features = bool(return_features or return_feat)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = self._align(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._align(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._align(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.seg_head(d1)

        if not return_features and not return_sdf:
            return logits

        outputs = [logits]

        if return_features:
            feat = self.feature_head(d1)
            outputs.append(feat)

        if return_sdf:
            sdf = self.sdf_head(d1)
            outputs.append(sdf)

        return tuple(outputs)