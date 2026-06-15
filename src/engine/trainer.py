from __future__ import annotations

import csv
from contextlib import nullcontext
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
    sdm_loss,
)

from src.utils.metrics import compute_binary_metrics, SegMetrics
from src.models.discriminator import SegDiscriminator


# =========================
# EMA
# =========================

@torch.no_grad()
def update_ema(student, teacher, m: float):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(m).add_(s.data * (1 - m))


def entropy(p: torch.Tensor):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# =========================
# Trainer (TMI stable core)
# =========================

class MeanTeacherTrainer:

    def __init__(self, student, cfg: dict):

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.student_aux = deepcopy(student).to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        self.optim = Adam(
            list(self.student.parameters()) +
            list(self.student_aux.parameters()),
            lr=float(cfg["train"]["lr"]),
        )

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=int(cfg["train"]["epochs"]),
        )

        self.epochs = int(cfg["train"]["epochs"])
        self.ema_m = float(cfg["train"].get("ema_momentum", 0.99))

        self.use_amp = bool(cfg["train"].get("use_amp", True))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.lambda_ssl = float(cfg["loss"]["lambda_ssl"])
        self.lambda_cps = float(cfg["loss"]["lambda_cps"])
        self.lambda_sdm = float(cfg["loss"]["lambda_sdm"])

    # =========================
    # uncertainty-aware weight
    # =========================

    def _uncertainty_weight(self, p):
        ent = entropy(p)
        w = torch.exp(-ent)
        return w / (w.mean() + 1e-6)

    # =========================
    # train
    # =========================

    def fit(self, loaders, out_dir: str):

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):

            self.student.train()
            self.student_aux.train()

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

                amp = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

                with amp:

                    s_l = self.student(x_l)[0]
                    s2_l = self.student_aux(x_l)[0]

                    sup = supervised_loss(s_l, y_l) + supervised_loss(s2_l, y_l)

                    with torch.no_grad():
                        t_u = self.teacher(x_u)[0]
                        p_u = torch.sigmoid(t_u)

                    s_u = self.student(x_u)[0]
                    s2_u = self.student_aux(x_u)[0]

                    w = self._uncertainty_weight(p_u)

                    unsup = unsupervised_loss(s_u, p_u, weight=w)
                    unsup2 = unsupervised_loss(s2_u, p_u, weight=w)

                    cps = cps_loss(s_u, s2_u)
                    sdm = sdm_loss(s_l, y_l)

                    loss = (
                        sup
                        + self.lambda_ssl * (unsup + unsup2)
                        + self.lambda_cps * cps
                        + self.lambda_sdm * sdm
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                update_ema(self.student, self.teacher, self.ema_m)

            self.scheduler.step()

            torch.save(self.student.state_dict(), out / f"epoch_{epoch}.pt")

        torch.save(self.student.state_dict(), out / "last.pt")