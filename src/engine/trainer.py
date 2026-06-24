from __future__ import annotations

import csv
import copy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from src.losses.contrastive import prototype_contrast_loss
from src.losses.seg_losses import (
    spectral_consistency_loss,
    supervised_loss,
    unsupervised_loss,
)
from src.utils.frequency import frequency_filter
from src.utils.metrics import compute_binary_metrics


class MeanTeacherTrainer:
    def __init__(self, model: torch.nn.Module, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = model.to(self.device)
        self.teacher = copy.deepcopy(model).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()

        lr = float(cfg.train.get("lr", 1e-4))
        self.optimizer = Adam(self.student.parameters(), lr=lr)

        self.epochs = int(cfg.train.get("epochs", 200))
        self.ema_momentum = float(cfg.train.get("ema_momentum", 0.99))
        self.grad_clip = float(cfg.train.get("grad_clip", 1.0))
        self.use_amp = bool(cfg.train.get("use_amp", True)) and self.device.type == "cuda"

        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs),
            eta_min=float(cfg.train.get("min_lr", 1e-6)),
        )

        self.lambda_unsup = float(cfg.loss.get("lambda_unsup", 0.3))
        self.lambda_spec = float(cfg.loss.get("lambda_spec", 0.05))
        self.lambda_proto = float(cfg.loss.get("lambda_proto", 0.05))

        self.prototype_memory: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def update_teacher(self) -> None:
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.ema_momentum).add_(s_param.data.detach(), alpha=1.0 - self.ema_momentum)

    def _next_unlabeled(self, unlabeled_iter, unlabeled_loader):
        try:
            return next(unlabeled_iter), unlabeled_iter
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            return next(unlabeled_iter), unlabeled_iter

    def train_step(self, batch_l, batch_u) -> float:
        self.student.train()
        self.teacher.eval()

        x_l = batch_l["image"].to(self.device, non_blocking=True).float()
        y_l = batch_l["label"].to(self.device, non_blocking=True).float()
        x_u = batch_u["image"].to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            s_l, feat_l = self.student(x_l, return_features=True)
            s_u, _ = self.student(x_u, return_features=True)

            loss_sup = supervised_loss(s_l, y_l)

        with torch.no_grad():
            t_u = self.teacher(x_u)
            if isinstance(t_u, tuple):
                t_u = t_u[0]
            pseudo = torch.sigmoid(t_u.detach()).float()
            pseudo = frequency_filter(pseudo)

        loss_unsup = unsupervised_loss(s_u, pseudo)
        loss_spec = spectral_consistency_loss(s_u, t_u.detach())
        loss_proto = prototype_contrast_loss(feat_l, y_l, self.prototype_memory)

        loss = (
            loss_sup
            + self.lambda_unsup * loss_unsup
            + self.lambda_spec * loss_spec
            + self.lambda_proto * loss_proto
        )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0

        self.scaler.scale(loss).backward()

        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.update_teacher()

        return float(loss.detach().cpu().item())

    def fit(self, loaders: dict, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        history_path = out / "history.csv"
        best_dice = -1.0

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_f1"])

            for epoch in range(1, self.epochs + 1):
                unlabeled_iter = iter(loaders["unlabeled"])

                total_loss = 0.0
                steps = 0

                for batch_l in loaders["labeled"]:
                    batch_u, unlabeled_iter = self._next_unlabeled(unlabeled_iter, loaders["unlabeled"])
                    loss = self.train_step(batch_l, batch_u)

                    total_loss += loss
                    steps += 1

                train_loss = total_loss / max(1, steps)
                metrics = self.evaluate(loaders["val"])
                self.scheduler.step()

                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        metrics.dice,
                        metrics.iou,
                        metrics.precision,
                        metrics.recall,
                        metrics.f1,
                    ]
                )
                f.flush()

                ckpt = {
                    "student": self.student.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                }

                torch.save(ckpt, out / "last.pt")

                if metrics.dice > best_dice:
                    best_dice = metrics.dice
                    ckpt["best_dice"] = best_dice
                    torch.save(ckpt, out / "best.pt")

                print(
                    f"[Epoch {epoch:03d}/{self.epochs}] "
                    f"loss={train_loss:.4f} dice={metrics.dice:.4f} iou={metrics.iou:.4f}"
                )

    @torch.no_grad()
    def evaluate(self, loader):
        self.teacher.eval()

        totals = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "minority_f1": 0.0,
            "hd95": 0.0,
        }
        n = 0

        threshold = float(self.cfg.inference.get("threshold", 0.5))

        for batch in loader:
            x = batch["image"].to(self.device).float()
            y = batch["label"].to(self.device).float()

            logits = self.teacher(x)
            if isinstance(logits, tuple):
                logits = logits[0]

            m = compute_binary_metrics(logits.float(), y.float(), threshold=threshold)

            totals["dice"] += float(m.dice)
            totals["iou"] += float(m.iou)
            totals["precision"] += float(m.precision)
            totals["recall"] += float(m.recall)
            totals["f1"] += float(m.f1)
            totals["minority_f1"] += float(m.minority_f1)
            totals["hd95"] += float(m.hd95)
            n += 1

        from src.utils.metrics import SegMetrics

        n = max(1, n)
        return SegMetrics(
            totals["dice"] / n,
            totals["iou"] / n,
            totals["precision"] / n,
            totals["recall"] / n,
            totals["f1"] / n,
            totals["minority_f1"] / n,
            totals["hd95"] / n,
        )