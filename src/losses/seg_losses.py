from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    if x.shape[2:] == ref.shape[2:]:
        return x

    kwargs = {"size": ref.shape[2:], "mode": mode}

    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False

    return F.interpolate(x, **kwargs)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    logits = logits.float()
    target = target.float()

    if logits.shape[2:] != target.shape[2:]:
        target = _resize_like(target, logits, mode="nearest")

    prob = torch.sigmoid(logits)

    dims = tuple(range(2, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def supervised_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    target = target.float()

    if logits.shape[2:] != target.shape[2:]:
        target = _resize_like(target, logits, mode="nearest")

    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)

    return bce + dice


def unsupervised_loss(logits: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    pseudo = pseudo.detach().float().clamp(0.0, 1.0)

    if logits.shape[2:] != pseudo.shape[2:]:
        pseudo = _resize_like(pseudo, logits, mode="trilinear").clamp(0.0, 1.0)

    prob = torch.sigmoid(logits)
    return F.mse_loss(prob, pseudo)


def spectral_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    max_size: int = 64,
) -> torch.Tensor:
    """
    AMP-safe spectral consistency.

    Important:
        cuFFT half precision fails for non-power-of-two sizes.
        Therefore this function always casts to fp32 and downsamples
        before FFT for memory stability.
    """
    s = student_logits.float()
    t = teacher_logits.detach().float()

    if s.shape[2:] != t.shape[2:]:
        t = _resize_like(t, s, mode="trilinear")

    spatial = s.shape[2:]
    target_size = tuple(min(int(v), max_size) for v in spatial)

    if spatial != target_size:
        s = F.interpolate(s, size=target_size, mode="trilinear", align_corners=False)
        t = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)

    with torch.amp.autocast("cuda", enabled=False):
        sf = torch.fft.fftn(torch.sigmoid(s.float()), dim=tuple(range(2, s.ndim))).abs()
        tf = torch.fft.fftn(torch.sigmoid(t.float()), dim=tuple(range(2, t.ndim))).abs()

        sf = torch.log1p(sf)
        tf = torch.log1p(tf)

        return F.mse_loss(sf, tf)


def entropy_loss(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.float().clamp(eps, 1.0 - eps)
    ent = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))
    return ent.mean()