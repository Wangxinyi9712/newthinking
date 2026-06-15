from __future__ import annotations

import csv
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses.seg_losses import (
    adaptive_tau_from_quantile,
    cps_loss,
    dynamic_pseudo_weight,
    feature_consistency_loss,
    fuse_pseudo_with_reliability,
    minority_sensitive_loss,
    sdm_loss,
    structural_loss,
    supervised_loss,
    teacher_prob_with_temperature,
    unsupervised_loss,
)

from utils.metrics import compute_binary_metrics


# =========================
# utilities
# =========================

def _safe(x):
    return float(torch.nan_to_num(x).detach().cpu())


def entropy_map(p: torch.Tensor, eps: float = 1e-6):
    p = p.clamp(eps, 1.0)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def mutual_information_like(p: torch.Tensor):
    # simplified MI proxy
    ent = entropy_map(p)
    return ent - torch.mean(ent)


def spectral_consistency_loss(x1: torch.Tensor, x2: torch.Tensor):
    """
    FFT amplitude consistency loss
    """
    def fft_amp(x):
        f = torch.fft.fftn(x.float(), dim=tuple(range(2, x.ndim)))
        return torch.abs(f)

    a1 = fft_amp(x1)
    a2 = fft_amp(x2)

    a1 = F.normalize(a1, dim=1)
    a2 = F.normalize(a2, dim=1)

    return F.l1_loss(a1, a2)


# =========================
# EMA (uncertainty-aware)
# =========================

@torch.no_grad()
def update_teacher(student, teacher, uncertainty, k=5.0):
    """
    alpha(x) = sigmoid(k * U)
    """
    alpha = torch.sigmoid(k * uncertainty.mean())

    for ts, ss in zip(teacher.parameters(), student.parameters()):
        ts.data.mul_(alpha).add_(ss.data * (1 - alpha))


# =========================
# Trainer
# =========================

class MeanTeacherTrainer:

    def __init__(self, student, cfg):

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        self.optim = Adam(self.student.parameters(), lr=cfg["train"]["lr"])

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("min_lr", 1e-6),
        )

        self.epochs = cfg["train"]["epochs"]
        self.use_amp = cfg["train"].get("use_amp", True)

    # =========================
    # reliability (INFO THEORY)
    # =========================

    def compute_reliability(self, p_student, p_teacher):

        ent = entropy_map(p_student)
        mi = mutual_information_like(p_student)

        divergence = F.l1_loss(p_student, p_teacher, reduction="none").mean(dim=1, keepdim=True)

        U = ent + divergence - mi
        U = torch.clamp(U, 0, 1)

        return 1 - U, U  # reliability, uncertainty

    # =========================
    # train epoch
    # =========================

    def train_epoch(self, loaders, epoch):

        self.student.train()
        self.teacher.eval()

        unlabeled_iter = iter(loaders["unlabeled"])

        total = 0.0

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

            with torch.cuda.amp.autocast(enabled=self.use_amp):

                s_l, _ = self.student(x_l)
                t_u, _ = self.teacher(x_u)
                s_u, _ = self.student(x_u)

                # supervised
                loss_sup = supervised_loss(s_l, y_l)

                # pseudo
                tau = adaptive_tau_from_quantile(t_u, q=0.65)
                pseudo_w = dynamic_pseudo_weight(t_u, tau=tau)[0]

                # reliability (INFO)
                rel, U = self.compute_reliability(
                    torch.sigmoid(s_u),
                    torch.sigmoid(t_u),
                )

                fused = fuse_pseudo_with_reliability(pseudo_w, rel)

                loss_unsup = unsupervised_loss(
                    s_u,
                    torch.sigmoid(t_u),
                    tau=tau,
                    fused_weight=fused,
                )

                # CPS
                loss_cps = cps_loss(s_u, s_u)

                # structural
                loss_struct = structural_loss(s_l, y_l)

                # SDM
                loss_sdm = sdm_loss(s_l, y_l)

                # spectral consistency (REPLACES GAN)
                loss_spec = spectral_consistency_loss(s_l, s_u)

                loss = (
                    loss_sup
                    + loss_unsup
                    + 0.3 * loss_cps
                    + 0.1 * loss_struct
                    + 0.1 * loss_sdm
                    + 0.2 * loss_spec
                )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # EMA update (uncertainty-aware)
            update_teacher(self.student, self.teacher, U)

            total += _safe(loss)

            pbar.set_postfix(loss=total / (pbar.n + 1))

        self.scheduler.step()
        return total / len(loaders["labeled"])

    # =========================
    # evaluation
    # =========================

    @torch.no_grad()
    def evaluate(self, loader):

        self.teacher.eval()

        total = 0.0

        for batch in loader:

            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            logits, _ = self.teacher(x)

            m = compute_binary_metrics(logits, y)

            total += m.dice

        return total / len(loader)

    # =========================
    # fit
    # =========================

    def fit(self, loaders, out_dir):

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        best = 0

        for epoch in range(self.epochs):

            loss = self.train_epoch(loaders, epoch)
            dice = self.evaluate(loaders["val"])

            if dice > best:
                best = dice
                torch.save(self.student.state_dict(), Path(out_dir) / "best.pt")

            print(f"[Epoch {epoch}] loss={loss:.4f} dice={dice:.4f}")