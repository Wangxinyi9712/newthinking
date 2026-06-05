from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss

_HD_LOSS = HausdorffDTLoss(alpha=2.0)


def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    target = target.float()
    inter = (probs * target).sum(dim=tuple(range(2, probs.ndim)))
    den = probs.sum(dim=tuple(range(2, probs.ndim))) + target.sum(dim=tuple(range(2, probs.ndim)))
    out = 1 - ((2 * inter + eps) / (den + eps)).mean()
    return _safe(out)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    p_t = probs * target + (1 - probs) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - p_t).pow(gamma) * bce
    return _safe(focal.mean())


def supervised_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, target.float())
    return _safe(ce + dice_loss(logits, target))


def dynamic_pseudo_weight(teacher_probs: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    tp = teacher_probs.clamp(0.0, 1.0)
    confidence = torch.maximum(tp, 1.0 - tp) if tp.shape[1] == 1 else tp.max(dim=1, keepdim=True).values
    weights = torch.sigmoid((confidence - tau) * 10.0)
    valid_mask = (confidence > tau).float()
    return _safe(weights), _safe(valid_mask)


def _normalize_map(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x = _safe(x)
    dims = tuple(range(2, x.ndim))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return _safe((x - x_min) / (x_max - x_min + eps))


def _entropy_map(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(1e-6, 1 - 1e-6)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return _normalize_map(ent)


def _consistency_map(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    return _normalize_map((student_probs - teacher_probs).abs())


def _ood_map(x_u: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, x_u.ndim))
    mu = x_u.mean(dim=dims, keepdim=True)
    std = x_u.std(dim=dims, keepdim=True).clamp_min(1e-6)
    z = ((x_u - mu).abs() / std).mean(dim=1, keepdim=True)
    return _normalize_map(z)


def reliability_components(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    x_u: torch.Tensor,
    student_feat: torch.Tensor | None = None,
    teacher_feat: torch.Tensor | None = None,
    temporal_teacher_probs: torch.Tensor | None = None,
    bank_mean: torch.Tensor | None = None,
    bank_var: torch.Tensor | None = None,
    enable_ood: bool = True,
    enable_consistency: bool = True,
) -> dict[str, torch.Tensor]:
    confidence = 1.0 - _normalize_map(torch.minimum(teacher_probs, 1.0 - teacher_probs))
    entropy = 1.0 - _entropy_map(teacher_probs)
    consistency = 1.0 - _consistency_map(student_probs, teacher_probs) if enable_consistency else torch.ones_like(confidence)
    if enable_ood:
        ood = 1.0 - _ood_map(x_u)
    else:
        ood = torch.ones_like(confidence)

    ones = torch.ones_like(confidence)
    return {
        "confidence_map": _safe(confidence),
        "entropy_map": _safe(entropy),
        "consistency_map": _safe(consistency),
        "ood_map": _safe(ood),
        "feature_distance_map": ones,
        "feature_embedding_map": ones,
        "gradient_uncertainty_map": ones,
        "temporal_consistency_map": ones,
        "transformer_feature_map": ones,
    }


def fuse_pseudo_with_reliability(
    pseudo_weight: torch.Tensor,
    reliability: torch.Tensor,
    gate_mlp: torch.nn.Module | None = None,
    mode: str = "convex",
    alpha: float = 0.7,
) -> torch.Tensor:
    pw = _safe(pseudo_weight).clamp(0.0, 1.0)
    rel = _safe(reliability).clamp(0.0, 1.0)
    if mode == "learnable" and gate_mlp is not None:
        gate_in = torch.cat([pw, rel], dim=1)
        fused = torch.sigmoid(gate_mlp(gate_in))
    elif mode == "multiply":
        fused = pw * rel
    else:
        fused = alpha * pw + (1 - alpha) * rel
    return _safe(fused).clamp(0.0, 1.0)


def unsupervised_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    tau: float = 0.65,
    fused_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pseudo = teacher_probs.detach().clamp(0.0, 1.0)
    _, valid_mask = dynamic_pseudo_weight(pseudo, tau)
    weights = fused_weight if fused_weight is not None else torch.ones_like(pseudo)
    weights = weights.clamp(1e-3, 1.0)

    ce = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none")
    weighted = _safe(ce * weights * valid_mask)
    denom = (weights * valid_mask).sum().clamp_min(1.0)
    return _safe(weighted.sum() / denom)


def minority_dice_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    target = target.float()
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + 1e-7) / (den + 1e-7)
    weighted = class_weights.to(logits.device).view(1, -1) * (1 - dice)
    return _safe(weighted.mean())


def minority_sensitive_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        weight=class_weights.to(logits.device).view(1, -1, *([1] * (logits.ndim - 2))),
    )
    return _safe(weighted_bce + focal_loss(logits, target, alpha=focal_alpha, gamma=focal_gamma) + minority_dice_loss(logits, target, class_weights))


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    grads = []
    for dim in range(2, x.ndim):
        fwd = torch.roll(x, shifts=-1, dims=dim)
        grads.append((fwd - x).abs())
    return _safe(torch.stack(grads, dim=0).mean(dim=0))


def structural_loss(
    logits: torch.Tensor,
    prior: torch.Tensor,
    hd_weight: float = 0.3,
    fg_weight: float = 2.0,
    topo_weight: float = 0.1,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    prior = prior.float().clamp(0.0, 1.0)

    pred_edge = _gradient_magnitude(probs)
    prior_edge = _gradient_magnitude(prior)

    fg_mask = (prior > 0.5).float()
    bg_mask = 1.0 - fg_mask
    weighted_edge = ((pred_edge - prior_edge).abs() * (fg_weight * fg_mask + bg_mask)).mean()

    hd_term = _HD_LOSS(probs, prior)
    return _safe(weighted_edge + hd_weight * hd_term + topo_weight * (pred_edge - prior_edge).abs().mean())


def feature_consistency_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    return _safe(F.mse_loss(student_feat, teacher_feat.detach()))