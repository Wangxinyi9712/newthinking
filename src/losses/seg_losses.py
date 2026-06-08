from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss

_HD_LOSS = HausdorffDTLoss(alpha=2.0)


def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def teacher_prob_with_temperature(logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    return torch.sigmoid(logits.float() / max(float(temperature), 1e-3)).clamp(0.0, 1.0)


def adaptive_tau_from_quantile(teacher_probs: torch.Tensor, q: float = 0.70, min_tau: float = 0.50, max_tau: float = 0.90) -> float:
    conf = torch.maximum(teacher_probs, 1.0 - teacher_probs).detach().float().reshape(-1)
    if conf.numel() == 0:
        return min_tau
    tau = torch.quantile(conf, q).item()
    return float(max(min_tau, min(max_tau, tau)))


def dynamic_pseudo_weight(teacher_probs: torch.Tensor, tau: float):
    tp = teacher_probs.float().clamp(0.0, 1.0)
    confidence = torch.maximum(tp, 1.0 - tp)
    w = torch.sigmoid((confidence - tau) * 10.0)
    m = (confidence > tau).float()
    return _safe(w), _safe(m)


def _normalize_map(x: torch.Tensor, eps: float = 1e-7):
    x = _safe(x.float())
    dims = tuple(range(2, x.ndim))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return _safe((x - mn) / (mx - mn + eps))


def reliability_components(student_probs: torch.Tensor, teacher_probs: torch.Tensor, x_u: torch.Tensor, enable_ood: bool = True, enable_consistency: bool = True):
    sp = student_probs.float()
    tp = teacher_probs.float()

    conf = 1.0 - _normalize_map(torch.minimum(tp, 1.0 - tp))
    p = tp.clamp(1e-6, 1 - 1e-6)
    ent = 1.0 - _normalize_map(-(p * torch.log(p) + (1 - p) * torch.log(1 - p)))
    cons = 1.0 - _normalize_map((sp - tp).abs()) if enable_consistency else torch.ones_like(conf)

    if enable_ood:
        xu = x_u.float()
        dims = tuple(range(2, xu.ndim))
        mu = xu.mean(dim=dims, keepdim=True)
        std = xu.std(dim=dims, keepdim=True).clamp_min(1e-6)
        ood = 1.0 - _normalize_map(((xu - mu).abs() / std).mean(dim=1, keepdim=True))
    else:
        ood = torch.ones_like(conf)

    ones = torch.ones_like(conf)
    return {
        "confidence_map": conf, "entropy_map": ent, "consistency_map": cons, "ood_map": ood,
        "feature_distance_map": ones, "feature_embedding_map": ones, "gradient_uncertainty_map": ones,
        "temporal_consistency_map": ones, "transformer_feature_map": ones,
    }


def fuse_pseudo_with_reliability(pseudo_weight: torch.Tensor, reliability: torch.Tensor, gate_mlp=None, mode: str = "convex", alpha: float = 0.75):
    pw = _safe(pseudo_weight).clamp(0.0, 1.0)
    r = _safe(reliability).clamp(0.0, 1.0)
    if mode == "learnable" and gate_mlp is not None:
        out = torch.sigmoid(gate_mlp(torch.cat([pw, r], dim=1)))
    elif mode == "multiply":
        out = pw * r
    else:
        out = alpha * pw + (1.0 - alpha) * r
    return _safe(out).clamp(0.0, 1.0)


def supervised_loss(logits: torch.Tensor, target: torch.Tensor):
    y = target.float().clamp(0.0, 1.0)
    z = logits.float()
    ce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z).clamp(0.0, 1.0)
    inter = (p * y).sum(dim=tuple(range(2, p.ndim)))
    den = p.sum(dim=tuple(range(2, p.ndim))) + y.sum(dim=tuple(range(2, p.ndim)))
    dice = 1.0 - ((2.0 * inter + 1e-7) / (den + 1e-7)).mean()
    return _safe(ce + dice)


def unsupervised_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, tau: float = 0.65, fused_weight=None, soft_gate=None, cvar_ratio: float = 0.20):
    pseudo = teacher_probs.detach().float().clamp(0.0, 1.0)
    z = student_logits.float()

    _, valid = dynamic_pseudo_weight(pseudo, tau)
    w = fused_weight.float() if fused_weight is not None else torch.ones_like(pseudo)
    if soft_gate is not None:
        w = w * soft_gate.float()
    w = w.clamp(1e-3, 1.0)

    ce = F.binary_cross_entropy_with_logits(z, pseudo, reduction="none")
    voxel = _safe(ce * w * valid)

    denom = (w * valid).sum().clamp_min(1.0)
    base = voxel.sum() / denom

    fg = (pseudo > 0.5).float()
    bg = 1.0 - fg

    def cvar_part(v, m, ratio):
        vec = (v * m).reshape(-1)
        vec = vec[vec > 0]
        if vec.numel() == 0:
            return torch.zeros((), device=v.device)
        k = max(1, int(vec.numel() * ratio))
        return torch.topk(vec, k=k, largest=True).values.mean()

    cvar_fg = cvar_part(voxel, fg, cvar_ratio)
    cvar_bg = cvar_part(voxel, bg, cvar_ratio)
    cvar = 0.6 * cvar_fg + 0.4 * cvar_bg
    return _safe(0.65 * base + 0.35 * cvar)


def minority_sensitive_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
    z = logits.float()
    y = target.float().clamp(0.0, 1.0)
    w = class_weights.to(z.device).float().view(1, -1, *([1] * (z.ndim - 2)))

    wbce = F.binary_cross_entropy_with_logits(z, y, weight=w)
    p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    at = focal_alpha * y + (1 - focal_alpha) * (1 - y)
    focal = (at * (1 - pt).pow(focal_gamma) * bce).mean()
    return _safe(wbce + focal)


def _gradient_magnitude(x: torch.Tensor):
    gs = []
    for d in range(2, x.ndim):
        gs.append((torch.roll(x, -1, dims=d) - x).abs())
    return _safe(torch.stack(gs, dim=0).mean(dim=0))


def structural_loss(logits: torch.Tensor, prior: torch.Tensor, hd_weight: float = 0.25, fg_weight: float = 2.0, topo_weight: float = 0.08, smooth_edge: bool = True):
    p = torch.sigmoid(logits.float()).clamp(0.0, 1.0)
    y = prior.float().clamp(0.0, 1.0)

    pe = _gradient_magnitude(p)
    ye = _gradient_magnitude(y)

    if smooth_edge:
        if p.ndim == 5:
            pe = F.avg_pool3d(pe, 3, 1, 1)
            ye = F.avg_pool3d(ye, 3, 1, 1)
        else:
            pe = F.avg_pool2d(pe, 3, 1, 1)
            ye = F.avg_pool2d(ye, 3, 1, 1)

    fg = (y > 0.5).float()
    bg = 1.0 - fg
    edge = ((pe - ye).abs() * (fg_weight * fg + bg)).mean()
    hd = _HD_LOSS(p, y)

    if p.ndim == 5:
        p_s = F.avg_pool3d(p, 3, 1, 1)
        y_s = F.avg_pool3d(y, 3, 1, 1)
    else:
        p_s = F.avg_pool2d(p, 3, 1, 1)
        y_s = F.avg_pool2d(y, 3, 1, 1)
    topo = (p_s - y_s).abs().mean()

    return _safe(edge + hd_weight * hd + topo_weight * topo)


def feature_consistency_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor):
    return _safe(F.mse_loss(student_feat.float(), teacher_feat.detach().float()))