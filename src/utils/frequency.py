from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def frequency_filter(x: torch.Tensor, max_size: int = 64, keep_ratio: float = 0.5) -> torch.Tensor:
    """
    AMP-safe low-frequency pseudo-label filter.

    Args:
        x: probability map [B, C, D, H, W]
    """
    if x.ndim != 5:
        raise ValueError(f"frequency_filter expects 5D [B,C,D,H,W], got {x.shape}")

    original_size = x.shape[2:]

    prob = x.detach().float().clamp(0.0, 1.0)

    target_size = tuple(min(int(v), max_size) for v in original_size)

    if original_size != target_size:
        prob_small = F.interpolate(prob, size=target_size, mode="trilinear", align_corners=False)
    else:
        prob_small = prob

    with torch.amp.autocast("cuda", enabled=False):
        fft = torch.fft.fftn(prob_small.float(), dim=tuple(range(2, prob_small.ndim)))

        shifted = torch.fft.fftshift(fft, dim=tuple(range(2, prob_small.ndim)))

        _, _, d, h, w = shifted.shape
        kd = max(1, int(d * keep_ratio / 2))
        kh = max(1, int(h * keep_ratio / 2))
        kw = max(1, int(w * keep_ratio / 2))

        mask = torch.zeros_like(shifted.real)
        cd, ch, cw = d // 2, h // 2, w // 2
        mask[:, :, cd - kd : cd + kd + 1, ch - kh : ch + kh + 1, cw - kw : cw + kw + 1] = 1.0

        filtered = shifted * mask
        filtered = torch.fft.ifftshift(filtered, dim=tuple(range(2, prob_small.ndim)))
        out = torch.fft.ifftn(filtered, dim=tuple(range(2, prob_small.ndim))).real

        out = out.clamp(0.0, 1.0)

    if original_size != target_size:
        out = F.interpolate(out, size=original_size, mode="trilinear", align_corners=False).clamp(0.0, 1.0)

    return out