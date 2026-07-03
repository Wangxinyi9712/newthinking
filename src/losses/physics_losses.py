from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _ensure_5d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"{name} must be 5D [B,C,D,H,W], got shape={tuple(x.shape)}")
    return x


def _match_spatial(
    x: torch.Tensor,
    ref: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    if x.shape[2:] == ref.shape[2:]:
        return x

    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[2:], mode="nearest")

    return F.interpolate(
        x,
        size=ref.shape[2:],
        mode=mode,
        align_corners=False if mode in {"trilinear", "bilinear"} else None,
    )


def spatial_gradient_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward finite difference gradients for 3D tensor.

    Input:
        x: [B,C,D,H,W]

    Output:
        dz, dy, dx, each with the same shape as x
    """
    _ensure_5d(x, "x")

    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))

    return dz, dy, dx


def edge_map_from_image(
    image: torch.Tensor,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute edge-aware diffusion coefficient:

        g(|grad I|) = exp(- beta * |grad I|^2)

    Input:
        image: [B,C,D,H,W]

    Output:
        g: [B,1,D,H,W]
    """
    image = _ensure_5d(image.float(), "image")

    gray = image.mean(dim=1, keepdim=True)

    mean = gray.mean(dim=(2, 3, 4), keepdim=True)
    std = gray.std(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
    gray = (gray - mean) / std

    dz, dy, dx = spatial_gradient_3d(gray)
    grad_mag2 = dz.pow(2) + dy.pow(2) + dx.pow(2)

    g = torch.exp(-float(beta) * grad_mag2)
    return g.clamp(0.0, 1.0)


@torch.no_grad()
def edge_aware_reaction_diffusion(
    pseudo: torch.Tensor,
    image: torch.Tensor,
    steps: int = 3,
    tau: float = 0.15,
    beta: float = 4.0,
    kappa: float = 0.2,
    smooth_kernel: int = 3,
) -> torch.Tensor:
    """
    Edge-aware reaction-diffusion pseudo-label refinement.

    Pseudo label is viewed as a concentration field q.
    The update is:

        q_{t+1} = q_t + tau * [ g(I) * (AvgPool(q_t)-q_t)
                                + kappa * q_t(1-q_t)(q_t-0.5) ]

    where:
        g(I) suppresses diffusion around image edges;
        reaction term sharpens probabilities toward 0/1.

    Input:
        pseudo: [B,1,D,H,W], teacher probability
        image:  [B,C,D,H,W], input image

    Output:
        refined pseudo label, [B,1,D,H,W]
    """
    pseudo = _ensure_5d(pseudo.float(), "pseudo").clamp(0.0, 1.0)
    image = _ensure_5d(image.float(), "image")

    q = pseudo.clone()
    g = edge_map_from_image(image, beta=beta)

    k = int(smooth_kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    pad = k // 2

    for _ in range(max(0, int(steps))):
        smooth = F.avg_pool3d(q, kernel_size=k, stride=1, padding=pad)
        diffusion = g * (smooth - q)
        reaction = q * (1.0 - q) * (q - 0.5)

        q = q + float(tau) * (diffusion + float(kappa) * reaction)
        q = q.clamp(0.0, 1.0)

    return q


def reliability_from_entropy(
    pseudo: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Entropy-based reliability weight.

        H(p) = -p log p - (1-p) log(1-p)
        r = exp(-H / T)

    Output:
        reliability in [exp(-log2/T), 1]
    """
    pseudo = _ensure_5d(pseudo.float(), "pseudo").clamp(eps, 1.0 - eps)

    entropy = -pseudo * torch.log(pseudo) - (1.0 - pseudo) * torch.log(1.0 - pseudo)
    entropy = entropy / math.log(2.0)

    t = max(float(temperature), eps)
    reliability = torch.exp(-entropy / t)

    return reliability.clamp(0.0, 1.0)


def weighted_mse_consistency(
    student_logits: torch.Tensor,
    pseudo: torch.Tensor,
    reliability: Optional[torch.Tensor] = None,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Reliability-weighted student-teacher consistency.

    Input:
        student_logits: [B,1,D,H,W]
        pseudo:         [B,1,D,H,W]
        reliability:    [B,1,D,H,W] or None
    """
    student_logits = _ensure_5d(student_logits, "student_logits")
    pseudo = _ensure_5d(pseudo.float(), "pseudo")
    pseudo = _match_spatial(pseudo, student_logits, mode="trilinear").clamp(0.0, 1.0)

    prob = torch.sigmoid(student_logits.float())

    if reliability is None:
        weight = torch.ones_like(pseudo)
    else:
        weight = _match_spatial(reliability.float(), student_logits, mode="trilinear")
        weight = weight.clamp(0.0, 1.0)

    if confidence_threshold > 0:
        conf = torch.maximum(pseudo, 1.0 - pseudo)
        weight = weight * (conf >= float(confidence_threshold)).float()

    loss = (prob - pseudo).pow(2) * weight
    denom = weight.sum().clamp_min(1.0)

    return loss.sum() / denom


def phase_field_loss(
    logits: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """
    Phase-field regularization:

        L = epsilon * |grad p|^2 + 1/epsilon * p^2(1-p)^2

    It encourages smooth boundaries and discourages uncertain probabilities.
    """
    logits = _ensure_5d(logits, "logits")

    p = torch.sigmoid(logits.float())
    dz, dy, dx = spatial_gradient_3d(p)

    grad_energy = dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()
    double_well = (p.pow(2) * (1.0 - p).pow(2)).mean()

    eps = max(float(epsilon), 1e-6)
    return eps * grad_energy + (1.0 / eps) * double_well


def volume_calibration_loss(
    student_logits: torch.Tensor,
    pseudo: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    OT-inspired volume calibration.

    Lightweight volume version:

        L_vol = | mean(sigmoid(student_logits)) - mean(target_probability) |

    Use pseudo for unlabeled data, target_mask for labeled/mixed data.
    """
    student_logits = _ensure_5d(student_logits, "student_logits")

    p = torch.sigmoid(student_logits.float())

    if pseudo is not None:
        target = _match_spatial(pseudo.float(), student_logits, mode="trilinear").clamp(0.0, 1.0)
    elif target_mask is not None:
        target = _match_spatial(target_mask.float(), student_logits, mode="nearest").clamp(0.0, 1.0)
    else:
        return student_logits.new_tensor(0.0)

    v_pred = p.mean(dim=(1, 2, 3, 4))
    v_tgt = target.mean(dim=(1, 2, 3, 4))

    return torch.abs(v_pred - v_tgt).mean()


def _signed_distance_numpy(mask_np, max_dist: float):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    mask_np = mask_np.astype(bool)

    if mask_np.sum() == 0:
        return np.full(mask_np.shape, fill_value=max_dist, dtype="float32")

    if mask_np.sum() == mask_np.size:
        return np.full(mask_np.shape, fill_value=-max_dist, dtype="float32")

    outside = distance_transform_edt(~mask_np)
    inside = distance_transform_edt(mask_np)

    sdf = outside - inside
    sdf = np.clip(sdf, -max_dist, max_dist).astype("float32")

    return sdf


def signed_distance_target(
    mask: torch.Tensor,
    max_dist: float = 20.0,
) -> torch.Tensor:
    """
    Compute signed distance field target from binary mask.

    Convention:
        inside foreground: negative distance
        outside foreground: positive distance

    Input:
        mask: [B,1,D,H,W]

    Output:
        sdf: [B,1,D,H,W]
    """
    mask = _ensure_5d(mask.float(), "mask")

    try:
        import numpy as np  # noqa: F401
        import scipy  # noqa: F401
    except Exception:
        # Fallback when scipy is not installed.
        # This still gives a boundary-related target but is less precise.
        return (1.0 - 2.0 * (mask > 0.5).float()) * float(max_dist)

    device = mask.device
    dtype = mask.dtype

    mask_cpu = mask.detach().cpu().numpy()
    b, c, d, h, w = mask_cpu.shape

    out = []

    for bi in range(b):
        channels = []
        for ci in range(c):
            sdf = _signed_distance_numpy(mask_cpu[bi, ci] > 0.5, max_dist=float(max_dist))
            channels.append(sdf)
        out.append(channels)

    import numpy as np

    sdf_np = np.asarray(out, dtype="float32")
    sdf = torch.from_numpy(sdf_np).to(device=device, dtype=dtype)

    return sdf


def sdf_regression_loss(
    pred_sdf: torch.Tensor,
    target_mask: torch.Tensor,
    max_dist: float = 20.0,
    tau: float = 2.0,
) -> torch.Tensor:
    """
    SDF boundary loss.

        L_sdf = | tanh(pred_sdf / tau) - tanh(target_sdf / tau) |

    pred_sdf is the output of the model's SDF head.
    """
    pred_sdf = _ensure_5d(pred_sdf.float(), "pred_sdf")
    target_mask = _ensure_5d(target_mask.float(), "target_mask")

    target_mask = _match_spatial(target_mask, pred_sdf, mode="nearest")
    target_sdf = signed_distance_target(target_mask, max_dist=float(max_dist))

    t = max(float(tau), 1e-6)

    pred = torch.tanh(pred_sdf / t)
    target = torch.tanh(target_sdf / t)

    return torch.mean(torch.abs(pred - target))