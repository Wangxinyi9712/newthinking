from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class SegMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    minority_f1: float
    hd95: float
    boundary_dice: float = 0.0
    assd: float = 0.0
    volume_error: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _match_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.shape[2:] == logits.shape[2:]:
        return target

    return F.interpolate(target.float(), size=logits.shape[2:], mode="nearest")


def _surface_distances(pred_np, gt_np):
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    pred_np = pred_np.astype(bool)
    gt_np = gt_np.astype(bool)

    if pred_np.sum() == 0 and gt_np.sum() == 0:
        return np.array([0.0], dtype="float32"), 0.0

    if pred_np.sum() == 0 or gt_np.sum() == 0:
        diag = float(np.linalg.norm(pred_np.shape))
        return np.array([diag], dtype="float32"), diag

    structure = generate_binary_structure(3, 1)

    pred_surface = pred_np ^ binary_erosion(pred_np, structure=structure, border_value=0)
    gt_surface = gt_np ^ binary_erosion(gt_np, structure=structure, border_value=0)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        diag = float(np.linalg.norm(pred_np.shape))
        return np.array([diag], dtype="float32"), diag

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype("float32")

    if distances.size == 0:
        return np.array([0.0], dtype="float32"), 0.0

    assd = float(distances.mean())
    return distances, assd


def _boundary_dice_single(pred_np, gt_np, tolerance: int = 2) -> float:
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    pred_np = pred_np.astype(bool)
    gt_np = gt_np.astype(bool)

    if pred_np.sum() == 0 and gt_np.sum() == 0:
        return 1.0

    if pred_np.sum() == 0 or gt_np.sum() == 0:
        return 0.0

    structure = generate_binary_structure(3, 1)

    pred_surface = pred_np ^ binary_erosion(pred_np, structure=structure, border_value=0)
    gt_surface = gt_np ^ binary_erosion(gt_np, structure=structure, border_value=0)

    pred_band = binary_dilation(pred_surface, structure=structure, iterations=int(tolerance))
    gt_band = binary_dilation(gt_surface, structure=structure, iterations=int(tolerance))

    pred_hit = pred_surface & gt_band
    gt_hit = gt_surface & pred_band

    denom = pred_surface.sum() + gt_surface.sum()

    if denom == 0:
        return 1.0

    return float((pred_hit.sum() + gt_hit.sum()) / denom)


def _distance_metrics_batch(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    """
    Return:
        hd95, assd, boundary_dice
    """
    try:
        import scipy  # noqa: F401
    except Exception:
        return 0.0, 0.0, 0.0

    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    b = pred_np.shape[0]

    hd95_values = []
    assd_values = []
    bd_values = []

    for i in range(b):
        p = pred_np[i, 0]
        g = target_np[i, 0]

        distances, assd = _surface_distances(p, g)

        import numpy as np

        hd95 = float(np.percentile(distances, 95))
        bd = _boundary_dice_single(p, g, tolerance=2)

        hd95_values.append(hd95)
        assd_values.append(float(assd))
        bd_values.append(float(bd))

    return (
        float(sum(hd95_values) / max(1, len(hd95_values))),
        float(sum(assd_values) / max(1, len(assd_values))),
        float(sum(bd_values) / max(1, len(bd_values))),
    )


@torch.no_grad()
def compute_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> SegMetrics:
    """
    Compute binary 3D segmentation metrics.

    Input:
        logits: [B,1,D,H,W]
        target: [B,1,D,H,W]
    """
    if logits.ndim != 5:
        raise ValueError(f"logits must be 5D [B,1,D,H,W], got {tuple(logits.shape)}")

    if target.ndim != 5:
        raise ValueError(f"target must be 5D [B,1,D,H,W], got {tuple(target.shape)}")

    target = _match_target(target.float(), logits)
    prob = torch.sigmoid(logits.float())

    pred = (prob >= float(threshold)).float()
    gt = (target >= 0.5).float()

    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)

    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - gt_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * gt_flat).sum(dim=1)

    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()

    f1 = dice
    minority_f1 = f1

    gt_volume = gt_flat.sum(dim=1)
    pred_volume = pred_flat.sum(dim=1)
    volume_error = (torch.abs(pred_volume - gt_volume) / gt_volume.clamp_min(1.0)).mean().item()

    hd95, assd, boundary_dice = _distance_metrics_batch(pred, gt)

    return SegMetrics(
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        minority_f1=float(minority_f1),
        hd95=float(hd95),
        boundary_dice=float(boundary_dice),
        assd=float(assd),
        volume_error=float(volume_error),
    )