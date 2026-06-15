from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim):
    return nn.Conv3d if dim == 3 else nn.Conv2d


# =========================================================
# SE BLOCK
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8, dim=3):
        super().__init__()
        Conv = _conv(dim)
        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool3d(1) if dim == 3 else nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            Conv(channels, hidden, 1),
            nn.ReLU(inplace=True),
            Conv(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        return x * self.fc(w)


# =========================================================
# Spatial Attention
# =========================================================
class SpatialAttention(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        Conv = _conv(dim)
        self.conv = Conv(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# =========================================================
# MultiScale Attention
# =========================================================
class MultiScaleSkipAttention(nn.Module):
    def __init__(self, channels, dim=3):
        super().__init__()
        Conv = _conv(dim)

        self.conv3 = Conv(channels, channels, 3, padding=1)
        self.conv5 = Conv(channels, channels, 5, padding=2)

        self.se = SEBlock(channels, dim=dim)
        self.sa = SpatialAttention(dim=dim)

        self.fuse = Conv(channels * 2, channels, 1)

    def forward(self, x):
        y = torch.cat([self.conv3(x), self.conv5(x)], dim=1)
        y = self.fuse(y)
        return self.sa(self.se(y))


# =========================================================
# Transformer Encoder
# =========================================================
class LiteTransformerEncoder(nn.Module):
    def __init__(self, channels, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)

        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        spatial = x.shape[2:]

        tokens = x.flatten(2).transpose(1, 2)

        h = self.norm1(tokens)
        attn, _ = self.attn(h, h, h)
        tokens = tokens + attn
        tokens = tokens + self.mlp(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(b, c, *spatial)


# =========================================================
# UNCERTAINTY GATE
# =========================================================
class UncertaintyGate(nn.Module):
    def __init__(self, channels, dim=3):
        super().__init__()
        Conv = _conv(dim)

        self.net = nn.Sequential(
            Conv(1, max(4, channels // 8), 1),
            nn.GELU(),
            Conv(max(4, channels // 8), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        u = x.var(dim=1, keepdim=True, unbiased=False)
        g = self.net(u)
        return x * (1 - g)


# =========================================================
# 🔥 REQUIRED BY seg_model.py (FIX YOUR ERROR)
# =========================================================
class FrequencyEnhance(nn.Module):
    """
    TMI-stable spectral enhancement block
    """
    def __init__(self, channels, dim=3, alpha=0.1):
        super().__init__()
        Conv = _conv(dim)
        self.alpha = alpha
        self.proj = Conv(channels, channels, 1)

    def forward(self, x):
        x_f = x.float()

        dims = tuple(range(2, x.ndim))
        freq = torch.fft.fftn(x_f, dim=dims)
        amp = torch.abs(freq)

        if len(dims) == 3:
            low = F.avg_pool3d(amp, 3, 1, 1)
        else:
            low = F.avg_pool2d(amp, 3, 1, 1)

        low = self.proj(low)

        return (x_f + self.alpha * low).to(x.dtype)


# =========================================================
# PROJECTION HEAD
# =========================================================
class PrototypeProjectionHead(nn.Module):
    def __init__(self, in_ch, proj_ch=64, dim=3):
        super().__init__()
        Conv = _conv(dim)

        self.net = nn.Sequential(
            Conv(in_ch, in_ch, 1),
            nn.GELU(),
            Conv(in_ch, proj_ch, 1),
        )

    def forward(self, x):
        z = self.net(x.float())
        return F.normalize(z, dim=1)