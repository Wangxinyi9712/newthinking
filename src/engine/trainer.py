from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.losses.seg_losses import (
    supervised_loss,
    unsupervised_loss,
    cps_loss,
    spectral_consistency_loss,
    uncertainty_weight,
    entropy,
)

from src.utils.metrics import compute_binary_metrics, SegMetrics


# =========================================================
# UNCERTAINTY-AWARE EMA (TMI CORE CONTRIBUTION)
# =========================================================
@torch.no_grad()
def update_ema(student, teacher, x, m_base=0.99):

    # compute uncertainty from student prediction
    p = torch.sigmoid(student(x)[0].detach())
    u = entropy(p).mean()

    alpha = torch.sigmoid(-u).item()
    momentum = m_base * alpha

    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data * (1 - momentum))


class MeanTeacherTrainer:

    def __init__(self, student, cfg: dict):

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        train_cfg = cfg.get("train", {})
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})

        self.optim = Adam(
            self.student.parameters(),
            lr=train_cfg.get("lr", 1e-4)
        )

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=train_cfg.get("epochs", 100)
        )

        self.epochs = train_cfg.get("epochs", 100)

        self.use_amp = train_cfg.get("use_amp", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # loss weights
        self.lambda_ssl = loss_cfg.get("lambda_ssl", 1.0)
        self.lambda_cps = loss_cfg.get("lambda_cps", 0.3)
        self.lambda_spec = 0.15  # fixed TMI contribution

        self.threshold = cfg.get("inference", {}).get("threshold", 0.45)

    # =========================================================
    # TRAIN
    # =========================================================
    def fit(self, loaders, out_dir: str):

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):

            self.student.train()

            unlabeled_iter = iter(loaders["unlabeled"])

            pbar = tqdm(loaders["labeled"], desc=f"epoch {epoch}")

            for batch in pbar:

                try:
                    ub = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(loaders["unlabeled"])
                    ub = next(unlabeled_iter)

                x_l = batch["image"].to(self.device)
                y_l = batch["label"].to(self.device)
                x_u = ub["image"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):

                    # -------------------------
                    # supervised
                    # -------------------------
                    s_l = self.student(x_l)[0]
                    sup_loss = supervised_loss(s_l, y_l)

                    # -------------------------
                    # teacher pseudo label
                    # -------------------------
                    with torch.no_grad():
                        t_u = self.teacher(x_u)[0]
                        p_u = torch.sigmoid(t_u)

                    # -------------------------
                    # student unsupervised
                    # -------------------------
                    s_u = self.student(x_u)[0]

                    w = uncertainty_weight(p_u)
                    unsup_loss = unsupervised_loss(s_u, p_u, weight=w)

                    # -------------------------
                    # CPS
                    # -------------------------
                    cps = cps_loss(s_u, t_u)

                    # -------------------------
                    # Spectral consistency (KEY)
                    # -------------------------
                    spec = spectral_consistency_loss(x_l, x_u)

                    loss = (
                        sup_loss
                        + self.lambda_ssl * unsup_loss
                        + self.lambda_cps * cps
                        + self.lambda_spec * spec
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                # -------------------------
                # EMA update (uncertainty-aware)
                # -------------------------
                update_ema(self.student, self.teacher, x_u)

            self.scheduler.step()

            # save checkpoint
            torch.save(
                self.student.state_dict(),
                out / f"epoch_{epoch}.pt"
            )

        torch.save(self.student.state_dict(), out / "last.pt")

    # =========================================================
    # EVAL
    # =========================================================
    @torch.no_grad()
    def evaluate(self, loader):

        self.student.eval()

        agg = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        n = 0

        for batch in loader:

            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            logits = self.teacher(x)[0]
            m = compute_binary_metrics(logits, y, threshold=self.threshold)

            agg["dice"] += float(m.dice)
            agg["iou"] += float(m.iou)
            agg["precision"] += float(m.precision)
            agg["recall"] += float(m.recall)
            agg["f1"] += float(m.f1)

            n += 1

        for k in agg:
            agg[k] /= max(n, 1)

        return SegMetrics(
            agg["dice"],
            agg["iou"],
            agg["precision"],
            agg["recall"],
            agg["f1"],
            agg["f1"],
            0.0
        )