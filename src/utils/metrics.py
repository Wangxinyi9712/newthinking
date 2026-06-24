from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


@dataclass
class SegMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    minority_f1: float
    hd95: float


def _prepare_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)

    target = target.float()

    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")

    return (target > 0.5).float()


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / (den + eps)


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)

    structure = np.ones((3,) * mask.ndim, dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def _hd95_one(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return float(max(pred.shape))

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return float(max(pred.shape))

    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    distances_pred_to_target = dt_target[pred_surface]
    distances_target_to_pred = dt_pred[target_surface]

    distances = np.concatenate([distances_pred_to_target, distances_target_to_pred], axis=0)

    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))


def _hd95_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not _SCIPY_AVAILABLE:
        return 0.0

    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    values = []

    # shape: [B, C, ...]
    for b in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            values.append(_hd95_one(pred_np[b, c], target_np[b, c]))

    if not values:
        return 0.0

    return float(np.mean(values))


def compute_binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    logits = logits.float()
    target = _prepare_target(logits, target)

    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    # 当前是二分类肿瘤前景任务，minority_f1 使用前景 F1。
    minority_f1 = f1

    hd95 = _hd95_batch(pred, target)

    return SegMetrics(
        dice=float(dice.detach().cpu().item()),
        iou=float(iou.detach().cpu().item()),
        precision=float(precision.detach().cpu().item()),
        recall=float(recall.detach().cpu().item()),
        f1=float(f1.detach().cpu().item()),
        minority_f1=float(minority_f1.detach().cpu().item()),
        hd95=float(hd95),
    )