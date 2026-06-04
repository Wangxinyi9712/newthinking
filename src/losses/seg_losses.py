from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss


_HD_LOSS = HausdorffDTLoss(alpha=2.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum(dim=tuple(range(2, probs.ndim)))
    den = probs.sum(dim=tuple(range(2, probs.ndim))) + target.sum(dim=tuple(range(2, probs.ndim)))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - p_t).pow(gamma) * bce
    return focal.mean()


def supervised_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, target.float())
    return ce + dice_loss(logits, target)


def dynamic_pseudo_weight(teacher_probs: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_probs.shape[1] == 1:
        confidence = torch.maximum(teacher_probs, 1.0 - teacher_probs)
    else:
        confidence = teacher_probs.max(dim=1, keepdim=True).values
    weights = torch.sigmoid((confidence - tau) * 10.0)
    valid_mask = (confidence > tau).float()
    return weights, valid_mask


def _normalize_map(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dims = tuple(range(2, x.ndim))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


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


def feature_distance_map(student_feat: torch.Tensor, teacher_feat: torch.Tensor, mode: str = "mahalanobis") -> torch.Tensor:
    diff = student_feat - teacher_feat
    if mode == "mahalanobis":
        var = teacher_feat.var(dim=tuple(range(2, teacher_feat.ndim)), keepdim=True).clamp_min(1e-6)
        dist = (diff.pow(2) / var).mean(dim=1, keepdim=True)
    else:
        dist = diff.pow(2).mean(dim=1, keepdim=True)
    return _normalize_map(dist)


def feature_embedding_map(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    sim = F.cosine_similarity(student_feat, teacher_feat, dim=1, eps=1e-6).unsqueeze(1)
    return _normalize_map((sim + 1.0) * 0.5)


def gradient_uncertainty_map(probs: torch.Tensor) -> torch.Tensor:
    grads = []
    for d in range(2, probs.ndim):
        grads.append((torch.roll(probs, -1, dims=d) - probs).abs())
    grad_mag = torch.stack(grads, dim=0).mean(dim=0)
    return _normalize_map(grad_mag)


def temporal_consistency_map(
    teacher_probs: torch.Tensor,
    temporal_teacher_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if temporal_teacher_probs is None:
        return torch.ones_like(teacher_probs)
    return 1.0 - _normalize_map((teacher_probs - temporal_teacher_probs).abs())


def transformer_feature_map(feat: torch.Tensor) -> torch.Tensor:
    return _normalize_map(feat.mean(dim=1, keepdim=True))


def mahalanobis_ood_map(
    teacher_feat: torch.Tensor,
    bank_mean: torch.Tensor | None = None,
    bank_var: torch.Tensor | None = None,
) -> torch.Tensor:
    dims = tuple(range(2, teacher_feat.ndim))
    cur_mean = teacher_feat.mean(dim=dims, keepdim=True)
    if bank_mean is None or bank_var is None:
        var = teacher_feat.var(dim=dims, keepdim=True).clamp_min(1e-6)
        dist = ((teacher_feat - cur_mean).pow(2) / var).mean(dim=1, keepdim=True)
    else:
        var = bank_var.clamp_min(1e-6)
        dist = ((teacher_feat - bank_mean).pow(2) / var).mean(dim=1, keepdim=True)
    return _normalize_map(dist)


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
    temporal_consistency = temporal_consistency_map(teacher_probs, temporal_teacher_probs) if enable_consistency else torch.ones_like(confidence)

    if student_feat is not None and teacher_feat is not None:
        feat_dist = 1.0 - feature_distance_map(student_feat, teacher_feat, mode="mahalanobis")
        feat_embed = feature_embedding_map(student_feat, teacher_feat)
        tf_map = transformer_feature_map(teacher_feat)
    else:
        feat_dist = torch.ones_like(confidence)
        feat_embed = torch.ones_like(confidence)
        tf_map = torch.ones_like(confidence)

    if enable_ood and teacher_feat is not None:
        ood = 1.0 - mahalanobis_ood_map(teacher_feat, bank_mean=bank_mean, bank_var=bank_var)
    else:
        ood = 1.0 - _ood_map(x_u) if enable_ood else torch.ones_like(confidence)

    grad_unc = 1.0 - gradient_uncertainty_map(teacher_probs)

    return {
        "confidence_map": confidence,
        "entropy_map": entropy,
        "consistency_map": consistency,
        "ood_map": ood,
        "feature_distance_map": feat_dist,
        "feature_embedding_map": feat_embed,
        "gradient_uncertainty_map": grad_unc,
        "temporal_consistency_map": temporal_consistency,
        "transformer_feature_map": tf_map,
    }


def fuse_pseudo_with_reliability(
    pseudo_weight: torch.Tensor,
    reliability: torch.Tensor,
    gate_mlp: torch.nn.Module | None = None,
    mode: str = "learnable",
    alpha: float = 0.5,
) -> torch.Tensor:
    if mode == "learnable" and gate_mlp is not None:
        gate_in = torch.cat([pseudo_weight, reliability], dim=1)
        fused = torch.sigmoid(gate_mlp(gate_in))
    elif mode == "convex":
        fused = alpha * pseudo_weight + (1 - alpha) * reliability
    else:
        fused = pseudo_weight * reliability
    return fused.clamp(0.0, 1.0)


def unsupervised_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    tau: float = 0.7,
    fused_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pseudo = teacher_probs.detach()  # soft pseudo labels
    _, valid_mask = dynamic_pseudo_weight(teacher_probs, tau)
    weights = fused_weight if fused_weight is not None else torch.ones_like(teacher_probs)
    ce = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none")
    weighted = ce * weights * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def minority_dice_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + 1e-7) / (den + 1e-7)
    weighted = class_weights.to(logits.device).view(1, -1) * (1 - dice)
    return weighted.mean()


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
    return weighted_bce + focal_loss(logits, target, alpha=focal_alpha, gamma=focal_gamma) + minority_dice_loss(logits, target, class_weights)


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    grads = []
    for dim in range(2, x.ndim):
        fwd = torch.roll(x, shifts=-1, dims=dim)
        grads.append((fwd - x).abs())
    return torch.stack(grads, dim=0).mean(dim=0)


def topology_connectivity_loss(probs: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    pred_mass = probs.sum(dim=tuple(range(2, probs.ndim)))
    true_mass = prior.sum(dim=tuple(range(2, prior.ndim)))
    mass_term = (pred_mass - true_mass).abs().mean()

    pred_pool = F.avg_pool3d(probs, 3, stride=1, padding=1) if probs.ndim == 5 else F.avg_pool2d(probs, 3, stride=1, padding=1)
    true_pool = F.avg_pool3d(prior, 3, stride=1, padding=1) if prior.ndim == 5 else F.avg_pool2d(prior, 3, stride=1, padding=1)
    conn_term = (pred_pool - true_pool).abs().mean()
    return mass_term + conn_term


def structural_loss(
    logits: torch.Tensor,
    prior: torch.Tensor,
    hd_weight: float = 0.5,
    fg_weight: float = 2.0,
    topo_weight: float = 0.2,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    prior = prior.float()
    pred_edge = _gradient_magnitude(probs)
    prior_edge = _gradient_magnitude(prior)

    fg_mask = (prior > 0.5).float()
    bg_mask = 1.0 - fg_mask
    weighted_edge = ((pred_edge - prior_edge).abs() * (fg_weight * fg_mask + bg_mask)).mean()

    hd_term = _HD_LOSS(probs, prior)
    topo_term = topology_connectivity_loss(probs, prior)
    return weighted_edge + hd_weight * hd_term + topo_weight * topo_term


def feature_consistency_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_feat, teacher_feat.detach())