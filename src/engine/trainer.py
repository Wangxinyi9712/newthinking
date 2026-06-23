from __future__ import annotations

import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from src.models.bayes_teacher import BayesianTeacher
from src.losses.seg_losses import (
    supervised_loss,
    unsupervised_loss,
    spectral_consistency_loss,
)

from src.losses.contrastive import prototype_contrast_loss
from src.utils.frequency import frequency_filter


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.device = torch.device("cuda")

        self.student = model.to(self.device)
        self.teacher = BayesianTeacher(model).to(self.device)

        self.opt = Adam(self.student.parameters(), lr=1e-4)

        # AMP safe
        self.scaler = GradScaler("cuda")

        # EMA
        self.ema_base = cfg.train.get("ema", 0.99)

        # prototype memory
        self.memory = {}

    # -------------------------
    # EMA update (stable version)
    # -------------------------
    @torch.no_grad()
    def update_teacher(self, uncertainty=None):

        ema = self.ema_base

        if uncertainty is not None:
            # clamp stability
            u = torch.clamp(uncertainty.mean(), 0.0, 1.0)
            ema = self.ema_base * (1.0 - u.item() * 0.1)

        for t, s in zip(self.teacher.model.parameters(), self.student.parameters()):
            t.data.mul_(ema).add_(s.data, alpha=1 - ema)

    # -------------------------
    # MAIN TRAIN STEP
    # -------------------------
    def train_step(self, batch_l, batch_u):

        x_l = batch_l["image"].to(self.device)
        y_l = batch_l["label"].to(self.device)

        x_u = batch_u["image"].to(self.device)

        self.opt.zero_grad(set_to_none=True)

        with autocast("cuda"):

            # -------------------------
            # student forward
            # -------------------------
            s_l, feat_l = self.student(x_l, return_features=True)
            s_u, feat_u = self.student(x_u, return_features=True)

            # -------------------------
            # teacher forward (Bayesian)
            # -------------------------
            t_mean, t_var = self.teacher(x_u)

            # uncertainty
            uncertainty = torch.sigmoid(t_var).detach()

            # -------------------------
            # pseudo label (stable)
            # -------------------------
            pseudo = t_mean.detach() * (1.0 - uncertainty)

        # =========================
        # ⚠️ IMPORTANT: leave AMP region
        # =========================

        pseudo = frequency_filter(pseudo)

        # resize safety (VERY IMPORTANT for BraTS)
        if pseudo.shape != s_u.shape:
            pseudo = torch.nn.functional.interpolate(
                pseudo,
                size=s_u.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        # -------------------------
        # losses
        # -------------------------
        loss_sup = supervised_loss(s_l, y_l)

        loss_unsup = unsupervised_loss(s_u, pseudo)

        loss_spec = spectral_consistency_loss(s_u, t_mean)

        # prototype loss (safe guarded)
        try:
            loss_proto = prototype_contrast_loss(feat_l, y_l, self.memory)
        except Exception as e:
            print("[WARN] prototype loss skipped:", e)
            loss_proto = torch.tensor(0.0, device=self.device)

        loss = (
            loss_sup +
            0.5 * loss_unsup +
            0.1 * loss_spec +
            0.05 * loss_proto
        )

        # -------------------------
        # backward
        # -------------------------
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        # -------------------------
        # update teacher
        # -------------------------
        self.update_teacher(uncertainty)

        return loss.item()