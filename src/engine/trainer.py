from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses.seg_losses import (
    supervised_loss,
    unsupervised_loss,
    cps_loss,
    sdm_loss,
    minority_sensitive_loss,
    adaptive_tau_from_quantile,
    teacher_prob_with_temperature,
    dynamic_pseudo_weight,
)

from utils.metrics import compute_binary_metrics


# =========================================================
# TMI: INFORMATION THEORY UTILITIES
# =========================================================

def entropy(p: torch.Tensor):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def mutual_uncertainty(p_s, p_t):
    # disagreement proxy
    return torch.abs(p_s - p_t)


def spectral_consistency_loss(x1, x2):
    """
    Fourier consistency (TMI replacement for GAN)
    """
    f1 = torch.fft.fftn(x1.float(), dim=list(range(2, x1.ndim)))
    f2 = torch.fft.fftn(x2.float(), dim=list(range(2, x2.ndim)))

    return F.l1_loss(torch.abs(f1), torch.abs(f2))


# =========================================================
# EMA (UNCERTAINTY-AWARE)
# =========================================================

@torch.no_grad()
def update_teacher_uncertainty_ema(student, teacher, p_s, p_t, gamma=2.0):
    """
    θ_t = α(x)θ_t + (1-α(x))θ_s
    α(x)=sigmoid(γU)
    """
    u = mutual_uncertainty(p_s, p_t).mean(dim=(1, 2, 3, 4), keepdim=True)
    alpha = torch.sigmoid(gamma * u)

    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(alpha.mean()).add_(s_p.data * (1 - alpha.mean()))


# =========================================================
# TRAINER
# =========================================================

class MeanTeacherTrainer:

    def __init__(self, student, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        # optimizer
        self.optim = Adam(self.student.parameters(), lr=cfg["train"]["lr"])

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("min_lr", 1e-6),
        )

        self.epochs = cfg["train"]["epochs"]
        self.lambda_ssl = cfg["loss"]["lambda_ssl"]
        self.lambda_struct = cfg["loss"].get("lambda_struct", 0.1)

        self.teacher_temp = cfg["loss"].get("teacher_temperature", 1.5)

    # =====================================================
    def train_one_epoch(self, loaders, epoch):

        self.student.train()

        unlabeled_iter = iter(loaders["unlabeled"])

        running = 0.0

        for batch in tqdm(loaders["labeled"]):

            try:
                ub = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(loaders["unlabeled"])
                ub = next(unlabeled_iter)

            x_l = batch["image"].to(self.device)
            y_l = batch["label"].to(self.device)

            x_u = ub["image"].to(self.device)

            # ---------------- supervised ----------------
            s_l, _, _ = self.student(x_l, return_features=True)
            loss_sup = supervised_loss(s_l, y_l)

            # ---------------- teacher ----------------
            with torch.no_grad():
                t_u, _, _ = self.teacher(x_u, return_features=True)
                t_prob = torch.sigmoid(t_u)

            s_u, _, _ = self.student(x_u, return_features=True)
            s_prob = torch.sigmoid(s_u)

            # ---------------- pseudo label ----------------
            tau = adaptive_tau_from_quantile(
                t_prob, q=0.65, min_tau=0.5, max_tau=0.9
            )

            pseudo_w = dynamic_pseudo_weight(t_prob, tau=tau)

            # ---------------- INFORMATION UNCERTAINTY ----------------
            ent = entropy(s_prob)
            mu = mutual_uncertainty(s_prob, t_prob)

            uncertainty = (ent + mu).detach()

            # spectral consistency (replace GAN)
            loss_spec = spectral_consistency_loss(s_prob, t_prob)

            # ---------------- unsupervised ----------------
            loss_unsup = unsupervised_loss(
                s_u,
                t_prob,
                tau=tau,
                fused_weight=pseudo_w,
                soft_gate=torch.exp(-uncertainty),
            )

            # ---------------- cps ----------------
            loss_cps = cps_loss(s_u, s_u)

            # ---------------- structural ----------------
            loss_struct = sdm_loss(s_l, y_l)

            # ---------------- total loss ----------------
            loss = (
                loss_sup
                + self.lambda_ssl * loss_unsup
                + 0.3 * loss_cps
                + self.lambda_struct * loss_struct
                + 0.2 * loss_spec
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # =================================================
            # UNCERTAINTY-AWARE TEACHER UPDATE (KEY CONTRIBUTION)
            # =================================================
            with torch.no_grad():
                update_teacher_uncertainty_ema(
                    self.student,
                    self.teacher,
                    s_prob,
                    t_prob,
                    gamma=2.0,
                )

            running += loss.item()

        return running / len(loaders["labeled"])

    # =====================================================
    def evaluate(self, loader):

        self.student.eval()
        self.teacher.eval()

        metrics = []

        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            with torch.no_grad():
                logits, _, _ = self.teacher(x, return_features=True)

            m = compute_binary_metrics(logits, y)

            metrics.append(m.dice)

        return sum(metrics) / len(metrics)

    # =====================================================
    def fit(self, loaders, out_dir):

        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        best = 0

        for epoch in range(self.epochs):

            loss = self.train_one_epoch(loaders, epoch)
            self.scheduler.step()

            dice = self.evaluate(loaders["val"])

            print(f"[Epoch {epoch}] loss={loss:.4f} dice={dice:.4f}")

            if dice > best:
                best = dice
                torch.save(
                    {
                        "student": self.student.state_dict(),
                        "teacher": self.teacher.state_dict(),
                    },
                    out_dir / "best.pt",
                )

        torch.save(self.student.state_dict(), out_dir / "last.pt")