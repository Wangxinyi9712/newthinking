import torch


class SegMetrics:
    def __init__(self, dice, iou, precision, recall, f1, minority_f1, hd95):
        self.dice = float(dice)
        self.iou = float(iou)
        self.precision = float(precision)
        self.recall = float(recall)
        self.f1 = float(f1)
        self.minority_f1 = float(minority_f1)
        self.hd95 = float(hd95)


def compute_binary_metrics(logits, target, threshold=0.5):

    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = target.float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = dice

    return SegMetrics(
        dice, iou, precision, recall, f1, f1, torch.tensor(0.0)
    )