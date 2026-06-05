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


def _sanitize_prob(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return x.clamp(0.0, 1.0)


def _sanitize_mask(x: torch.Tensor) -> torch.Tensor:
    # 关键：target 必须二值化，避免 label 非0/1导致 tp>pred.sum()，从而 precision/dice > 1
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return (x > 0.5).float()


def _tp_fp_fn(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _sanitize_mask(pred)
    target = _sanitize_mask(target)
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    return tp, fp, fn


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = (num + EPS) / (den + EPS)
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.clamp(0.0, 1.0)


def _f1_from_binary(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = _tp_fp_fn(pred, target)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return f1.clamp(0.0, 1.0)


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
    pred = _sanitize_mask(pred)
    target = _sanitize_mask(target)

    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    tgt_np = target.detach().cpu().numpy().astype(np.uint8)

    vals: list[float] = []
    for b in range(pred_np.shape[0]):
        vals.append(_hd95_single(pred_np[b, 0], tgt_np[b, 0]))

    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return float("inf")
    return float(np.percentile(finite, 95))


def compute_binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> SegMetrics:
    probs = _sanitize_prob(torch.sigmoid(logits))
    pred = (probs > threshold).float()
    target = _sanitize_mask(target)

    tp, fp, fn = _tp_fp_fn(pred, target)
    inter = tp
    union = pred.sum() + target.sum() - inter

    dice = _safe_ratio(2 * inter, pred.sum() + target.sum())
    iou = _safe_ratio(inter, union)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    minority_idx = max(0, target.shape[1] - 1)
    pred_m = pred[:, minority_idx : minority_idx + 1]
    tgt_m = target[:, minority_idx : minority_idx + 1]
    minority_f1 = _f1_from_binary(pred_m, tgt_m)
    hd95 = compute_hd95(pred_m, tgt_m)

    return SegMetrics(
        float(dice.item()),
        float(iou.item()),
        float(precision.item()),
        float(recall.item()),
        float(f1.item()),
        float(minority_f1.item()),
        float(hd95),
    )