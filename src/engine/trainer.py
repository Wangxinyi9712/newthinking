from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from src.losses.contrastive import prototype_contrast_loss
from src.losses.physics_losses import (
    edge_aware_reaction_diffusion,
    phase_field_loss,
    reliability_from_entropy,
    sdf_regression_loss,
    volume_calibration_loss,
    weighted_mse_consistency,
)
from src.losses.seg_losses import (
    spectral_consistency_loss,
    supervised_loss,
)
from src.utils.frequency import frequency_filter
from src.utils.metrics import SegMetrics, compute_binary_metrics


class MeanTeacherTrainer:
    """
    Unified trainer for:

    Existing groups:
      A: Supervised U-Net
      B: Mean Teacher
      C: Mean Teacher + Frequency
      D: Mean Teacher + Prototype
      E: Full Method

    External baselines:
      F: nnU-Net-style supervised
      G: BCP-style
      H: UniMatch-style
      I: CorrMatch-style
      J: WT-BCP-style

    New cross-disciplinary methods:
      K: UniMatch + SDF
      L: UniMatch + PDE
      M: UniMatch + OT
      N: UniMatch + PDE + SDF + OT
      O: UniMatch + PDE + SDF + OT + Prototype
    """

    def __init__(self, model: torch.nn.Module, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = model.to(self.device)
        self.teacher = copy.deepcopy(model).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()

        self.optimizer = Adam(
            self.student.parameters(),
            lr=float(cfg.train.get("lr", 1e-4)),
        )

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

        self.group_name = str(cfg.loss.get("group_name", "unnamed"))
        self.baseline_method = str(cfg.loss.get("baseline_method", "mean_teacher")).lower()

        self.use_unlabeled = bool(cfg.loss.get("use_unlabeled", True))
        if self.baseline_method in {"supervised", "nnunet", "nnunet_style"}:
            self.use_unlabeled = False

        self.use_frequency_filter = bool(cfg.loss.get("use_frequency_filter", False))

        self.use_pde_refine = bool(cfg.loss.get("use_pde_refine", False)) or ("pde" in self.baseline_method)
        self.use_sdf_loss = bool(cfg.loss.get("use_sdf_loss", False)) or ("sdf" in self.baseline_method)
        self.use_phase_loss = bool(cfg.loss.get("use_phase_loss", False)) or ("phase" in self.baseline_method)
        self.use_ot_loss = bool(cfg.loss.get("use_ot_loss", False)) or ("ot" in self.baseline_method)

        self.use_strong_consistency = (
            bool(cfg.loss.get("use_strong_consistency", False))
            or ("unimatch" in self.baseline_method)
            or self.baseline_method in {
                "unimatch",
                "unimatch_sdf",
                "unimatch_pde",
                "unimatch_ot",
                "pde_sdf_ot_unimatch",
                "pde_sdf_ot_proto_unimatch",
            }
        )

        self.lambda_unsup = float(cfg.loss.get("lambda_unsup", 0.0))
        self.lambda_spec = float(cfg.loss.get("lambda_spec", 0.0))
        self.lambda_proto = float(cfg.loss.get("lambda_proto", 0.0))
        self.lambda_sdf = float(cfg.loss.get("lambda_sdf", 0.0))
        self.lambda_phase = float(cfg.loss.get("lambda_phase", 0.0))
        self.lambda_ot = float(cfg.loss.get("lambda_ot", 0.0))

        self.pseudo_threshold = float(cfg.loss.get("pseudo_threshold", 0.5))
        self.confidence_threshold = float(cfg.loss.get("confidence_threshold", 0.0))
        self.reliability_temperature = float(cfg.loss.get("reliability_temperature", 1.0))

        self.strong_noise_std = float(cfg.loss.get("strong_noise_std", 0.03))
        self.strong_intensity_scale = float(cfg.loss.get("strong_intensity_scale", 0.20))
        self.strong_dropout_prob = float(cfg.loss.get("strong_dropout_prob", 0.10))
        self.unimatch_num_strong = int(cfg.loss.get("unimatch_num_strong", 2))

        self.corr_alpha = float(cfg.loss.get("corr_alpha", 0.65))
        self.corr_smooth_kernel = int(cfg.loss.get("corr_smooth_kernel", 7))

        self.wt_low_kernel = int(cfg.loss.get("wt_low_kernel", 7))
        self.wt_high_weight = float(cfg.loss.get("wt_high_weight", 1.0))

        self.pde_steps = int(cfg.loss.get("pde_steps", 3))
        self.pde_tau = float(cfg.loss.get("pde_tau", 0.15))
        self.pde_beta = float(cfg.loss.get("pde_beta", 4.0))
        self.pde_kappa = float(cfg.loss.get("pde_kappa", 0.2))
        self.pde_smooth_kernel = int(cfg.loss.get("pde_smooth_kernel", 3))

        self.sdf_max_dist = float(cfg.loss.get("sdf_max_dist", 20.0))
        self.sdf_tau = float(cfg.loss.get("sdf_tau", 2.0))
        self.phase_epsilon = float(cfg.loss.get("phase_epsilon", 1.0))

        self.max_train_steps = int(cfg.train.get("max_train_steps", 0) or 0)
        self.max_val_batches = int(cfg.train.get("max_val_batches", 0) or 0)
        self.log_every = int(cfg.train.get("log_every", 10) or 10)

        self.prototype_memory: dict[int, torch.Tensor] = {}

        print(
            f"[Trainer] group={self.group_name} | method={self.baseline_method} | "
            f"device={self.device} | amp={self.use_amp} | epochs={self.epochs} | "
            f"use_unlabeled={self.use_unlabeled} | strong={self.use_strong_consistency} | "
            f"pde={self.use_pde_refine} | sdf={self.use_sdf_loss} | "
            f"phase={self.use_phase_loss} | ot={self.use_ot_loss} | "
            f"lambda_unsup={self.lambda_unsup} | lambda_sdf={self.lambda_sdf} | "
            f"lambda_phase={self.lambda_phase} | lambda_ot={self.lambda_ot} | "
            f"lambda_proto={self.lambda_proto}"
        )

    def _autocast_device(self) -> str:
        return "cuda" if self.device.type == "cuda" else "cpu"

    @torch.no_grad()
    def update_teacher(self) -> None:
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.ema_momentum).add_(
                s_param.data.detach(),
                alpha=1.0 - self.ema_momentum,
            )

    def _next_unlabeled(self, unlabeled_iter, unlabeled_loader):
        try:
            return next(unlabeled_iter), unlabeled_iter
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            return next(unlabeled_iter), unlabeled_iter

    @staticmethod
    def _match_spatial(x: torch.Tensor, ref: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x

        if mode == "nearest":
            return torch.nn.functional.interpolate(x, size=ref.shape[2:], mode="nearest")

        return torch.nn.functional.interpolate(
            x,
            size=ref.shape[2:],
            mode=mode,
            align_corners=False,
        )

    @staticmethod
    def _safe_kernel(k: int) -> int:
        k = int(k)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def _low_pass_3d(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        k = self._safe_kernel(kernel_size)
        pad = k // 2
        return torch.nn.functional.avg_pool3d(
            x.float(),
            kernel_size=k,
            stride=1,
            padding=pad,
        )

    def _strong_augment_pair(
        self,
        x: torch.Tensor,
        pseudo: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Strong tensor-space perturbation for UniMatch-style consistency.
        Geometric flips are applied to both image and pseudo label.
        Intensity/noise/dropout are applied only to the image.
        """
        out = x
        p = pseudo
        r = reliability

        if self.strong_intensity_scale > 0:
            b = out.shape[0]
            scale = 1.0 + (
                torch.rand((b, 1, 1, 1, 1), device=out.device) * 2.0 - 1.0
            ) * self.strong_intensity_scale
            shift = (
                torch.rand((b, 1, 1, 1, 1), device=out.device) * 2.0 - 1.0
            ) * (self.strong_intensity_scale * 0.25)
            out = out * scale + shift

        if self.strong_noise_std > 0:
            out = out + torch.randn_like(out) * self.strong_noise_std

        if self.strong_dropout_prob > 0:
            keep = (torch.rand_like(out[:, :1]) > self.strong_dropout_prob).float()
            out = out * keep

        for dim in [2, 3, 4]:
            if torch.rand((), device=out.device) < 0.5:
                out = torch.flip(out, dims=[dim])
                p = torch.flip(p, dims=[dim])
                if r is not None:
                    r = torch.flip(r, dims=[dim])

        return out, p, r

    def _teacher_pseudo(
        self,
        x_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t_u = self.teacher(x_u)
            if isinstance(t_u, tuple):
                t_u = t_u[0]

            pseudo = torch.sigmoid(t_u.detach()).float()

            if self.use_frequency_filter:
                pseudo = frequency_filter(pseudo)

            if self.use_pde_refine:
                pseudo = edge_aware_reaction_diffusion(
                    pseudo=pseudo,
                    image=x_u,
                    steps=self.pde_steps,
                    tau=self.pde_tau,
                    beta=self.pde_beta,
                    kappa=self.pde_kappa,
                    smooth_kernel=self.pde_smooth_kernel,
                )

            reliability = reliability_from_entropy(
                pseudo,
                temperature=self.reliability_temperature,
            )

        return t_u, pseudo, reliability

    def _supervised_branch(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_l, feat_l, sdf_l = self.student(
            x_l,
            return_features=True,
            return_sdf=True,
        )

        loss_sup = supervised_loss(s_l, y_l)

        if self.use_sdf_loss and self.lambda_sdf > 0:
            loss_sdf = sdf_regression_loss(
                pred_sdf=sdf_l,
                target_mask=y_l,
                max_dist=self.sdf_max_dist,
                tau=self.sdf_tau,
            )
        else:
            loss_sdf = s_l.new_tensor(0.0)

        if self.use_phase_loss and self.lambda_phase > 0:
            loss_phase = phase_field_loss(
                s_l,
                epsilon=self.phase_epsilon,
            )
        else:
            loss_phase = s_l.new_tensor(0.0)

        return s_l, feat_l, sdf_l, loss_sup, loss_sdf + loss_phase

    def _mean_teacher_unsup_loss(
        self,
        x_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_u, _ = self.student(x_u, return_features=True)
        t_u, pseudo, reliability = self._teacher_pseudo(x_u)

        loss_unsup = weighted_mse_consistency(
            student_logits=s_u,
            pseudo=pseudo,
            reliability=reliability,
            confidence_threshold=self.confidence_threshold,
        )

        if self.lambda_spec > 0:
            loss_spec = spectral_consistency_loss(s_u, t_u.detach())
        else:
            loss_spec = s_u.new_tensor(0.0)

        if self.use_ot_loss and self.lambda_ot > 0:
            loss_ot = volume_calibration_loss(s_u, pseudo=pseudo)
        else:
            loss_ot = s_u.new_tensor(0.0)

        return loss_unsup, loss_spec, loss_ot

    def _unimatch_unsup_loss(
        self,
        x_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pseudo, reliability = self._teacher_pseudo(x_u)

        total_unsup = x_u.new_tensor(0.0)
        total_ot = x_u.new_tensor(0.0)

        num = max(1, self.unimatch_num_strong)

        for _ in range(num):
            x_s, p_s, r_s = self._strong_augment_pair(x_u, pseudo, reliability)

            logits_s = self.student(x_s)
            if isinstance(logits_s, tuple):
                logits_s = logits_s[0]

            total_unsup = total_unsup + weighted_mse_consistency(
                student_logits=logits_s,
                pseudo=p_s,
                reliability=r_s,
                confidence_threshold=self.confidence_threshold,
            )

            if self.use_ot_loss and self.lambda_ot > 0:
                total_ot = total_ot + volume_calibration_loss(logits_s, pseudo=p_s)

        loss_unsup = total_unsup / float(num)
        loss_ot = total_ot / float(num)

        loss_spec = x_u.new_tensor(0.0)
        return loss_unsup, loss_spec, loss_ot

    def _bcp_unsup_loss(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        x_u: torch.Tensor,
        use_wavelet: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pseudo, _ = self._teacher_pseudo(x_u)

        b = min(x_l.shape[0], y_l.shape[0], x_u.shape[0], pseudo.shape[0])
        x_l = x_l[:b]
        y_l = y_l[:b]
        x_u = x_u[:b]
        pseudo = pseudo[:b]

        y_l = self._match_spatial(y_l, x_u, mode="nearest")
        pseudo = self._match_spatial(pseudo, x_u, mode="trilinear")

        fg = (y_l > self.pseudo_threshold).float()

        x_mix = x_u * (1.0 - fg) + x_l * fg
        y_mix = pseudo * (1.0 - fg) + y_l * fg

        if use_wavelet:
            low_u = self._low_pass_3d(x_u, self.wt_low_kernel)
            low_mix = self._low_pass_3d(x_mix, self.wt_low_kernel)
            high_mix = x_mix - low_mix
            x_mix = low_u + self.wt_high_weight * high_mix

        logits_mix = self.student(x_mix)
        if isinstance(logits_mix, tuple):
            logits_mix = logits_mix[0]

        loss_unsup = supervised_loss(logits_mix, y_mix)

        if self.use_ot_loss and self.lambda_ot > 0:
            loss_ot = volume_calibration_loss(logits_mix, pseudo=y_mix)
        else:
            loss_ot = logits_mix.new_tensor(0.0)

        loss_spec = logits_mix.new_tensor(0.0)
        return loss_unsup, loss_spec, loss_ot

    def _corrmatch_unsup_loss(
        self,
        x_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pseudo, reliability = self._teacher_pseudo(x_u)

        smooth = self._low_pass_3d(pseudo, self.corr_smooth_kernel)
        pseudo_corr = self.corr_alpha * pseudo + (1.0 - self.corr_alpha) * smooth
        pseudo_corr = pseudo_corr.clamp(0.0, 1.0)

        logits_u, _ = self.student(x_u, return_features=True)

        loss_unsup = weighted_mse_consistency(
            student_logits=logits_u,
            pseudo=pseudo_corr,
            reliability=reliability,
            confidence_threshold=self.confidence_threshold,
        )

        if self.use_ot_loss and self.lambda_ot > 0:
            loss_ot = volume_calibration_loss(logits_u, pseudo=pseudo_corr)
        else:
            loss_ot = logits_u.new_tensor(0.0)

        loss_spec = logits_u.new_tensor(0.0)
        return loss_unsup, loss_spec, loss_ot

    def _baseline_unsup_loss(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        x_u: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_unlabeled or x_u is None or self.lambda_unsup <= 0:
            z = x_l.new_tensor(0.0)
            return z, z, z

        method = self.baseline_method

        if method == "bcp":
            return self._bcp_unsup_loss(x_l, y_l, x_u, use_wavelet=False)

        if method == "wtbcp":
            return self._bcp_unsup_loss(x_l, y_l, x_u, use_wavelet=True)

        if method == "corrmatch":
            return self._corrmatch_unsup_loss(x_u)

        if self.use_strong_consistency:
            return self._unimatch_unsup_loss(x_u)

        return self._mean_teacher_unsup_loss(x_u)

    def train_step(self, batch_l, batch_u=None) -> float:
        self.student.train()
        self.teacher.eval()

        x_l = batch_l["image"].to(self.device, non_blocking=True).float()
        y_l = batch_l["label"].to(self.device, non_blocking=True).float()

        x_u = None
        if batch_u is not None:
            x_u = batch_u["image"].to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self._autocast_device(), enabled=self.use_amp):
            s_l, feat_l, _, loss_sup, loss_geom = self._supervised_branch(x_l, y_l)

        loss_unsup, loss_spec, loss_ot = self._baseline_unsup_loss(x_l, y_l, x_u)

        if self.lambda_proto > 0:
            loss_proto = prototype_contrast_loss(feat_l, y_l, self.prototype_memory)
        else:
            loss_proto = s_l.new_tensor(0.0)

        loss = (
            loss_sup
            + self.lambda_unsup * loss_unsup
            + self.lambda_spec * loss_spec
            + self.lambda_sdf * loss_geom
            + self.lambda_phase * s_l.new_tensor(0.0)
            + self.lambda_ot * loss_ot
            + self.lambda_proto * loss_proto
        )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            print("[WARN] non-finite loss detected, skip this step.")
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
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_dice",
                    "val_iou",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "val_minority_f1",
                    "val_hd95",
                    "val_boundary_dice",
                    "val_assd",
                    "val_volume_error",
                ]
            )

            for epoch in range(1, self.epochs + 1):
                unlabeled_iter = iter(loaders["unlabeled"])

                total_loss = 0.0
                steps = 0

                labeled_iterable = loaders["labeled"]

                if tqdm is not None:
                    labeled_iterable = tqdm(
                        loaders["labeled"],
                        desc=f"Epoch {epoch}/{self.epochs}",
                        total=len(loaders["labeled"]),
                    )

                for batch_l in labeled_iterable:
                    batch_u = None

                    if self.use_unlabeled:
                        batch_u, unlabeled_iter = self._next_unlabeled(
                            unlabeled_iter,
                            loaders["unlabeled"],
                        )

                    loss = self.train_step(batch_l, batch_u)

                    total_loss += loss
                    steps += 1

                    if tqdm is not None:
                        labeled_iterable.set_postfix(loss=f"{loss:.4f}")
                    elif steps % self.log_every == 0:
                        print(f"[Epoch {epoch}] step={steps} loss={loss:.4f}")

                    if self.max_train_steps > 0 and steps >= self.max_train_steps:
                        break

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
                        metrics.minority_f1,
                        metrics.hd95,
                        metrics.boundary_dice,
                        metrics.assd,
                        metrics.volume_error,
                    ]
                )
                f.flush()

                ckpt = {
                    "student": self.student.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": dict(self.cfg),
                }

                torch.save(ckpt, out / "last.pt")

                if metrics.dice > best_dice:
                    best_dice = metrics.dice
                    ckpt["best_dice"] = best_dice
                    torch.save(ckpt, out / "best.pt")

                print(
                    f"[Epoch {epoch:03d}/{self.epochs}] "
                    f"loss={train_loss:.4f} "
                    f"dice={metrics.dice:.4f} "
                    f"iou={metrics.iou:.4f} "
                    f"hd95={metrics.hd95:.4f} "
                    f"bdice={metrics.boundary_dice:.4f} "
                    f"assd={metrics.assd:.4f}"
                )

    @torch.no_grad()
    def evaluate(self, loader) -> SegMetrics:
        use_teacher = bool(self.cfg.inference.get("use_teacher_ema", True))
        model = self.teacher if use_teacher else self.student
        model.eval()

        totals = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "minority_f1": 0.0,
            "hd95": 0.0,
            "boundary_dice": 0.0,
            "assd": 0.0,
            "volume_error": 0.0,
        }

        n = 0
        threshold = float(self.cfg.inference.get("threshold", 0.5))

        for batch in loader:
            x = batch["image"].to(self.device, non_blocking=True).float()
            y = batch["label"].to(self.device, non_blocking=True).float()

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]

            m = compute_binary_metrics(logits.float(), y.float(), threshold=threshold)

            for k in totals:
                totals[k] += float(getattr(m, k))

            n += 1

            if self.max_val_batches > 0 and n >= self.max_val_batches:
                break

        n = max(1, n)

        return SegMetrics(
            dice=totals["dice"] / n,
            iou=totals["iou"] / n,
            precision=totals["precision"] / n,
            recall=totals["recall"] / n,
            f1=totals["f1"] / n,
            minority_f1=totals["minority_f1"] / n,
            hd95=totals["hd95"] / n,
            boundary_dice=totals["boundary_dice"] / n,
            assd=totals["assd"] / n,
            volume_error=totals["volume_error"] / n,
        )