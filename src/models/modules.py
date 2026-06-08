from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim: int):
    return nn.Conv3d if dim == 3 else nn.Conv2d


def _adapt_pool(dim: int):
    return nn.AdaptiveAvgPool3d if dim == 3 else nn.AdaptiveAvgPool2d


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, dim: int = 3):
        super().__init__()
        conv = _conv(dim)
        pool = _adapt_pool(dim)
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            pool(1),
            conv(channels, hidden, 1),
            nn.ReLU(inplace=True),
            conv(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self, dim: int = 3, kernel_size: int = 7):
        super().__init__()
        conv = _conv(dim)
        self.conv = conv(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class MultiScaleSkipAttention(nn.Module):
    def __init__(self, channels: int, dim: int = 3):
        super().__init__()
        conv = _conv(dim)
        groups = max(1, channels // 8)
        self.conv3 = conv(channels, channels, 3, padding=1, groups=groups)
        self.conv5 = conv(channels, channels, 5, padding=2, groups=groups)
        self.se = SEBlock(channels, dim=dim)
        self.sa = SpatialAttention(dim=dim)
        self.out = conv(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv3(x), self.conv5(x)], dim=1)
        y = self.out(y)
        return self.sa(self.se(y))


class LiteTransformerEncoder(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.view(b, c, -1).transpose(1, 2)
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + y
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(b, c, *spatial)


class UncertaintyGate(nn.Module):
    def __init__(self, channels: int, dim: int = 3):
        super().__init__()
        conv = _conv(dim)
        hidden = max(4, channels // 8)
        self.net = nn.Sequential(conv(1, hidden, 1), nn.GELU(), conv(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        unc = skip.var(dim=1, keepdim=True, unbiased=False)
        gate = self.net(unc)
        return skip * (1.0 - gate)


class FrequencyEnhance(nn.Module):
    def __init__(self, channels: int, dim: int = 3, alpha: float = 0.12):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        conv = _conv(dim)
        self.proj = conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        xf = x.float()
        spatial_dims = tuple(range(2, xf.ndim))
        freq = torch.fft.fftn(xf, dim=spatial_dims)
        amp = torch.abs(freq)

        if self.dim == 3:
            low = F.avg_pool3d(amp, 3, 1, 1)
        else:
            low = F.avg_pool2d(amp, 3, 1, 1)

        low = low / (low.amax(dim=spatial_dims, keepdim=True) + 1e-6)
        low = self.proj(low)
        out = xf + self.alpha * low
        return out.to(dtype=x_dtype)


class PrototypeProjectionHead(nn.Module):
    def __init__(self, in_ch: int, proj_ch: int = 64, dim: int = 3):
        super().__init__()
        conv = _conv(dim)
        self.net = nn.Sequential(conv(in_ch, in_ch, 1), nn.GELU(), conv(in_ch, proj_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.float())
        return F.normalize(z, p=2, dim=1)