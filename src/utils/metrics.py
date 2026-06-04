from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


EPS = 1e-7


@dataclass
class SegMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    minority_f1: float
    hd95: float


def _tp_fp_fn(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return tp, fp, fn


def _f1_from_binary(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = _tp_fp_fn(pred, target)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    return (2 * precision * recall + EPS) / (precision + recall + EPS)


def _hd95_single(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_pts = np.argwhere(pred_mask > 0)
    tgt_pts = np.argwhere(target_mask > 0)
    if len(pred_pts) == 0 and len(tgt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return float("inf")
    d1 = directed_hausdorff(pred_pts, tgt_pts)[0]
    d2 = directed_hausdorff(tgt_pts, pred_pts)[0]
    return float(max(d1, d2))


def compute_hd95(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    tgt_np = target.detach().cpu().numpy().astype(np.uint8)
    vals = []
    for b in range(pred_np.shape[0]):
        vals.append(_hd95_single(pred_np[b, 0], tgt_np[b, 0]))
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return float("inf")
    return float(np.percentile(finite, 95))


def compute_binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    tp, fp, fn = _tp_fp_fn(pred, target)
    inter = tp
    union = pred.sum() + target.sum() - inter

    dice = (2 * inter + EPS) / (pred.sum() + target.sum() + EPS)
    iou = (inter + EPS) / (union + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    f1 = (2 * precision * recall + EPS) / (precision + recall + EPS)

    minority_idx = target.shape[1] - 1
    minority_f1 = _f1_from_binary(pred[:, minority_idx : minority_idx + 1], target[:, minority_idx : minority_idx + 1])
    hd95 = compute_hd95(pred[:, minority_idx : minority_idx + 1], target[:, minority_idx : minority_idx + 1])

    return SegMetrics(float(dice), float(iou), float(precision), float(recall), float(f1), float(minority_f1), float(hd95))
