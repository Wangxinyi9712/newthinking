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
    adaptive_tau_from_quantile,
    cps_loss,
    minority_sensitive_loss,
    structural_loss,
    supervised_loss,
    unsupervised_loss,
)


# =========================================================
# Spectral Energy Prior (stable version)
# =========================================================
def spectral_energy(p1: torch.Tensor, p2: torch.Tensor):
    dims = tuple(range(2, p1.ndim))

    f1 = torch.fft.fftn(p1.float(), dim=dims)
    f2 = torch.fft.fftn(p2.float(), dim=dims)

    a1 = torch.abs(f1)
    a2 = torch.abs(f2)

    # stable frequency split
    low1 = F.avg_pool3d(a1, 3, 1, 1)
    low2 = F.avg_pool3d(a2, 3, 1, 1)

    high1 = a1 - low1
    high2 = a2 - low2

    return F.l1_loss(low1, low2) + 0.5 * F.l1_loss(high1, high2)


# =========================================================
# Uncertainty (simple + stable)
# =========================================================
def uncertainty(p1, p2):
    eps = 1e-6
    p1 = p1.clamp(eps, 1 - eps)
    p2 = p2.clamp(eps, 1 - eps)

    entropy = -(p1 * torch.log(p1) + (1 - p1) * torch.log(1 - p1))
    disagreement = (p1 - p2).abs()

    return (entropy + disagreement).mean()


# =========================================================
# EMA update (stable Bayesian form)
# =========================================================
@torch.no_grad()
def ema_update(student, teacher, u, base=0.99):
    u = float(torch.nan_to_num(u).detach().cpu())
    alpha = base / (1.0 + u)
    alpha = float(torch.clamp(torch.tensor(alpha), 0.90, 0.999))

    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(alpha).add_(s.data * (1 - alpha))


def _safe(x):
    return float(torch.nan_to_num(x).item())


# =========================================================
# Trainer
# =========================================================
class MeanTeacherTrainer:
    def __init__(self, student, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.student_aux = deepcopy(student).to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        self.optim = Adam(
            list(self.student.parameters()) + list(self.student_aux.parameters()),
            lr=float(cfg["train"]["lr"]),
        )

        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=int(cfg["train"]["epochs"]),
            eta_min=float(cfg["train"].get("min_lr", 1e-6)),
        )

        self.epochs = int(cfg["train"]["epochs"])
        self.grad_clip = float(cfg["train"]["grad_clip"])
        self.grad_accum = int(cfg["train"]["grad_accum_steps"])
        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # weights
        self.lambda_u = float(cfg["loss"]["lambda_ssl"])
        self.lambda_c = float(cfg["loss"]["lambda_cps"])
        self.lambda_s = float(cfg["loss"]["lambda_struct"])
        self.lambda_m = float(cfg["loss"]["lambda_minor"])
        self.lambda_f = 0.15
        self.lambda_freq = 0.25

    # =====================================================
    def fit(self, loaders, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        best = -1

        for epoch in range(self.epochs):
            loss = self.train_epoch(loaders, epoch)
            metric = self.evaluate(loaders["val"])

            if metric.dice > best:
                best = metric.dice
                torch.save(self.student.state_dict(), out / "best.pt")

        torch.save(self.student.state_dict(), out / "last.pt")

    # =====================================================
    def train_epoch(self, loaders, epoch):
        self.student.train()
        self.student_aux.train()

        unlabeled_iter = iter(loaders["unlabeled"])
        running = 0.0
        step = 0

        self.optim.zero_grad(set_to_none=True)

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

                s_l, _ = self.student(x_l, return_features=True)
                s2_l, _ = self.student_aux(x_l, return_features=True)

                sup = supervised_loss(s_l, y_l) + supervised_loss(s2_l, y_l)

                with torch.no_grad():
                    t_u, _ = self.teacher(x_u, return_features=True)
                    t_prob = torch.sigmoid(t_u)

                s_u, _ = self.student(x_u, return_features=True)
                s2_u, _ = self.student_aux(x_u, return_features=True)

                p_u = torch.sigmoid(s_u)
                p2_u = torch.sigmoid(s2_u)

                tau = adaptive_tau_from_quantile(t_prob)

                l_unsup = unsupervised_loss(s_u, t_prob, tau=tau)
                l_unsup2 = unsupervised_loss(s2_u, t_prob, tau=tau)

                cps = cps_loss(s_u, s2_u)

                minor = minority_sensitive_loss(
                    s_l, y_l,
                    torch.tensor([2.0], device=self.device)
                )

                struct = structural_loss(s_l, y_l)

                freq = spectral_energy(p_u, p2_u)

                unc = uncertainty(p_u, p2_u)

                loss = (
                               sup
                               + self.lambda_u * (l_unsup + l_unsup2)
                               + self.lambda_c * cps
                               + self.lambda_s * struct
                               + self.lambda_m * minor
                               + self.lambda_freq * freq
                       ) / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)

                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                ema_update(
                    self.student,
                    self.teacher,
                    u=unc,
                    base=0.99,
                )

            running += _safe(loss)
            step += 1

        self.scheduler.step()
        return running / max(1, step)

    # =====================================================
    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()

        d = 0
        n = 0

        for b in loader:
            x = b["image"].to(self.device)
            y = b["label"].to(self.device)

            logits, _ = self.student(x)
            from utils.metrics import compute_binary_metrics

            m = compute_binary_metrics(logits, y)

            d += float(m.dice)
            n += 1

        from utils.metrics import SegMetrics

        return SegMetrics(
            dice=d / max(1, n),
            iou=0, precision=0, recall=0, f1=0, minority_f1=0, hd95=0,
        )
