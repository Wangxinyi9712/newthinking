from __future__ import annotations

import csv
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# =========================
# FIXED IMPORT PATH (CRITICAL)
# =========================
from src.losses.seg_losses import (
    adaptive_tau_from_quantile,
    cps_loss,
    dynamic_pseudo_weight,
    feature_consistency_loss,
    fuse_pseudo_with_reliability,
    gan_discriminator_loss,
    gan_generator_loss,
    minority_sensitive_loss,
    reliability_components,
    sdm_loss,
    structural_loss,
    supervised_loss,
    teacher_prob_with_temperature,
    unsupervised_loss,
)

from src.models.discriminator import SegDiscriminator
from src.utils.metrics import compute_binary_metrics, SegMetrics


# =========================
# EMA update
# =========================

@torch.no_grad()
def update_ema(student, teacher, momentum: float):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))


def _safe_scalar(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0).item())


def _is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


# =========================
# Trainer (TMI version base)
# =========================

class MeanTeacherTrainer:
    def __init__(self, student: torch.nn.Module, cfg: dict):

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.student_aux = deepcopy(student).to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        dim = 3 if cfg["data"].get("dim", "3d") == "3d" else 2

        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        # reliability head (kept)
        self.reliability_mlp = torch.nn.Sequential(
            conv(9, 16, 1),
            torch.nn.GELU(),
            conv(16, 1, 1),
        ).to(self.device)

        self.fusion_gate_mlp = torch.nn.Sequential(
            conv(2, 8, 1),
            torch.nn.GELU(),
            conv(8, 1, 1),
        ).to(self.device)

        # =========================
        # GAN removed in TMI version
        # replaced by spectral consistency later
        # =========================
        self.use_adv = False
        self.discriminator = None

        self.optim_g = Adam(
            list(self.student.parameters())
            + list(self.student_aux.parameters())
            + list(self.reliability_mlp.parameters())
            + list(self.fusion_gate_mlp.parameters()),
            lr=float(cfg["train"]["lr"]),
        )

        self.scheduler = CosineAnnealingLR(
            self.optim_g,
            T_max=int(cfg["train"]["epochs"]),
            eta_min=float(cfg["train"].get("min_lr", 1e-6)),
        )

        self.epochs = int(cfg["train"]["epochs"])
        self.grad_clip = float(cfg["train"].get("grad_clip", 1.0))

        self.ema_m = float(cfg["train"].get("ema_momentum", 0.99))

        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # loss weights
        self.lambda_ssl = float(cfg["loss"].get("lambda_ssl", 1.0))
        self.lambda_minor = float(cfg["loss"].get("lambda_minor", 0.8))
        self.lambda_struct = float(cfg["loss"].get("lambda_struct", 0.08))
        self.lambda_feat = float(cfg["loss"].get("lambda_feat_consistency", 0.05))
        self.lambda_cps = float(cfg["loss"].get("lambda_cps", 0.3))
        self.lambda_sdm = float(cfg["loss"].get("lambda_sdm", 0.15))

        self.teacher_temp = float(cfg["loss"].get("teacher_temperature", 1.4))
        self.tau_quantile = float(cfg["loss"].get("tau_quantile", 0.65))
        self.base_tau = float(cfg["loss"].get("tau", 0.62))

    # =========================
    # EMA (unchanged but stable)
    # =========================
    def _update_teacher(self):
        update_ema(self.student, self.teacher, self.ema_m)

    # =========================
    # train
    # =========================
    def fit(self, loaders, out_dir: str):

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):

            self.student.train()
            self.student_aux.train()
            self.reliability_mlp.train()
            self.fusion_gate_mlp.train()

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

                amp_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

                with amp_ctx:

                    s_l, s_l_sdm, _ = self.student(x_l, return_features=True)
                    s2_l, s2_l_sdm, _ = self.student_aux(x_l, return_features=True)

                    l_sup = supervised_loss(s_l, y_l) + supervised_loss(s2_l, y_l)
                    l_sdm = sdm_loss(s_l_sdm, y_l) + sdm_loss(s2_l_sdm, y_l)

                    with torch.no_grad():
                        t_u, _, _ = self.teacher(x_u, return_features=True)
                        t_u = teacher_prob_with_temperature(t_u, self.teacher_temp)

                    s_u, _, _ = self.student(x_u, return_features=True)
                    s2_u, _, _ = self.student_aux(x_u, return_features=True)

                    tau = self.base_tau

                    rel = reliability_components(s_u, t_u, x_u)

                    stacked = torch.cat(list(rel.values()), dim=1)
                    reliability = torch.sigmoid(self.reliability_mlp(stacked))

                    pseudo_w, mask = dynamic_pseudo_weight(t_u, tau=tau)

                    l_unsup = unsupervised_loss(s_u, t_u, tau=tau, fused_weight=pseudo_w)
                    l_cps = cps_loss(s_u, s2_u, mask)

                    loss = (
                        l_sup
                        + self.lambda_ssl * l_unsup
                        + self.lambda_cps * l_cps
                        + self.lambda_minor * l_sdm
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim_g)
                self.scaler.update()
                self.optim_g.zero_grad(set_to_none=True)

                self._update_teacher()

            self.scheduler.step()

            torch.save(self.student.state_dict(), out / f"epoch_{epoch}.pt")

        torch.save(self.student.state_dict(), out / "last.pt")