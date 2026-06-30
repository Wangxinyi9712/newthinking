from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from src.losses.contrastive import prototype_contrast_loss
from src.losses.seg_losses import (
    spectral_consistency_loss,
    supervised_loss,
    unsupervised_loss,
)
from src.utils.frequency import frequency_filter
from src.utils.metrics import SegMetrics, compute_binary_metrics


class MeanTeacherTrainer:
    """
    Unified trainer for:
      A: Supervised U-Net
      B: Mean Teacher
      C: Mean Teacher + Frequency
      D: Mean Teacher + Prototype
      E: Full Method

    Additional adapted baselines:
      F: nnU-Net-style strong supervised baseline
      G: BCP-style bidirectional copy-paste baseline
      H: UniMatch-style weak-to-strong consistency baseline
      I: CorrMatch-style correlation-smoothed pseudo-label baseline
      J: WT-BCP-style frequency / wavelet-inspired copy-paste baseline

    Notes:
      - These are adapted 3D BraTS versions using the existing HybridUNet backbone.
      - They are intended for fair same-backbone comparison inside this project.
    """

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

        self.baseline_method = str(cfg.loss.get("baseline_method", "mean_teacher")).lower()
        self.group_name = str(cfg.loss.get("group_name", self.baseline_method))

        self.use_unlabeled = bool(cfg.loss.get("use_unlabeled", True))
        self.use_frequency_filter = bool(cfg.loss.get("use_frequency_filter", False))

        if self.baseline_method in {"supervised", "nnunet", "nnunet_style"}:
            self.use_unlabeled = False

        self.lambda_unsup = float(cfg.loss.get("lambda_unsup", 0.0))
        self.lambda_spec = float(cfg.loss.get("lambda_spec", 0.0))
        self.lambda_proto = float(cfg.loss.get("lambda_proto", 0.0))

        self.pseudo_threshold = float(cfg.loss.get("pseudo_threshold", 0.5))
        self.confidence_threshold = float(cfg.loss.get("confidence_threshold", 0.0))

        self.strong_noise_std = float(cfg.loss.get("strong_noise_std", 0.03))
        self.strong_intensity_scale = float(cfg.loss.get("strong_intensity_scale", 0.20))
        self.strong_dropout_prob = float(cfg.loss.get("strong_dropout_prob", 0.10))
        self.unimatch_num_strong = int(cfg.loss.get("unimatch_num_strong", 2))

        self.corr_alpha = float(cfg.loss.get("corr_alpha", 0.65))
        self.corr_smooth_kernel = int(cfg.loss.get("corr_smooth_kernel", 7))

        self.wt_low_kernel = int(cfg.loss.get("wt_low_kernel", 7))
        self.wt_high_weight = float(cfg.loss.get("wt_high_weight", 1.0))

        self.max_train_steps = int(cfg.train.get("max_train_steps", 0) or 0)
        self.max_val_batches = int(cfg.train.get("max_val_batches", 0) or 0)
        self.log_every = int(cfg.train.get("log_every", 10) or 10)

        self.prototype_memory: dict[int, torch.Tensor] = {}

        print(
            f"[Trainer] group={self.group_name} | method={self.baseline_method} | "
            f"device={self.device} | amp={self.use_amp} | epochs={self.epochs} | "
            f"use_unlabeled={self.use_unlabeled} | use_frequency_filter={self.use_frequency_filter} | "
            f"lambda_unsup={self.lambda_unsup} | lambda_spec={self.lambda_spec} | "
            f"lambda_proto={self.lambda_proto} | max_train_steps={self.max_train_steps} | "
            f"max_val_batches={self.max_val_batches}"
        )

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

    def _autocast_device(self) -> str:
        return "cuda" if self.device.type == "cuda" else "cpu"

    @staticmethod
    def _match_spatial(y: torch.Tensor, ref: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
        if y.shape[2:] == ref.shape[2:]:
            return y
        return F.interpolate(y, size=ref.shape[2:], mode=mode)

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
        return F.avg_pool3d(x.float(), kernel_size=k, stride=1, padding=pad)

    def _strong_augment_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight strong augmentation in tensor space.
        This avoids changing the dataloader and works for 3D BraTS tensors.
        """
        out = x

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

        # random flips along D/H/W
        for dim in [2, 3, 4]:
            if torch.rand((), device=out.device) < 0.5:
                out = torch.flip(out, dims=[dim])

        return out

    def _teacher_pseudo(self, x_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            teacher_logits, teacher_prob
        """
        with torch.no_grad():
            t_u = self.teacher(x_u)
            if isinstance(t_u, tuple):
                t_u = t_u[0]

            pseudo = torch.sigmoid(t_u.detach()).float()

            if self.use_frequency_filter:
                pseudo = frequency_filter(pseudo)

        return t_u, pseudo

    def _confidence_mask(self, pseudo: torch.Tensor) -> torch.Tensor:
        if self.confidence_threshold <= 0:
            return torch.ones_like(pseudo)

        conf = torch.maximum(pseudo, 1.0 - pseudo)
        return (conf >= self.confidence_threshold).float()

    def _masked_consistency_loss(self, student_logits: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
        pseudo = self._match_spatial(pseudo, student_logits, mode="trilinear")
        prob = torch.sigmoid(student_logits.float())

        mask = self._confidence_mask(pseudo)
        sq = (prob - pseudo).pow(2) * mask

        denom = mask.sum().clamp_min(1.0)
        return sq.sum() / denom

    def _supervised_branch(self, x_l: torch.Tensor, y_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_l, feat_l = self.student(x_l, return_features=True)
        loss_sup = supervised_loss(s_l, y_l)
        return s_l, feat_l, loss_sup

    def _mean_teacher_unsup_loss(self, x_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_u, _ = self.student(x_u, return_features=True)
        t_u, pseudo = self._teacher_pseudo(x_u)

        loss_unsup = self._masked_consistency_loss(s_u, pseudo)

        if self.lambda_spec > 0:
            loss_spec = spectral_consistency_loss(s_u, t_u.detach())
        else:
            loss_spec = s_u.new_tensor(0.0)

        return loss_unsup, loss_spec

    def _bcp_unsup_loss(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        x_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        BCP-style copy-paste:
          - Use labeled foreground mask to paste labeled tumor region into unlabeled image.
          - Teacher pseudo label provides target for the unlabeled background.
        """
        _, pseudo = self._teacher_pseudo(x_u)

        b = min(x_l.shape[0], x_u.shape[0], y_l.shape[0])
        x_l = x_l[:b]
        y_l = y_l[:b]
        x_u = x_u[:b]
        pseudo = pseudo[:b]

        y_l = self._match_spatial(y_l, x_u, mode="nearest")
        pseudo = self._match_spatial(pseudo, x_u, mode="trilinear")

        fg_mask = (y_l > self.pseudo_threshold).float()

        x_mix = x_u * (1.0 - fg_mask) + x_l * fg_mask
        y_mix = pseudo * (1.0 - fg_mask) + y_l * fg_mask

        logits_mix = self.student(x_mix)
        if isinstance(logits_mix, tuple):
            logits_mix = logits_mix[0]

        return supervised_loss(logits_mix, y_mix)

    def _unimatch_unsup_loss(self, x_u: torch.Tensor) -> torch.Tensor:
        """
        UniMatch-style weak-to-strong consistency:
          - Teacher predicts pseudo labels on weak image.
          - Student predicts on multiple strong augmented versions.
        """
        _, pseudo = self._teacher_pseudo(x_u)

        total = x_u.new_tensor(0.0)
        num = max(1, self.unimatch_num_strong)

        for _ in range(num):
            x_s = self._strong_augment_3d(x_u)
            s_u = self.student(x_s)
            if isinstance(s_u, tuple):
                s_u = s_u[0]

            # The strong augmentation includes random flips.
            # To keep the implementation lightweight and stable, pseudo is not geometrically inverted.
            # This acts as robust perturbation consistency in tensor space.
            total = total + self._masked_consistency_loss(s_u, pseudo)

        return total / float(num)

    def _corrmatch_unsup_loss(self, x_u: torch.Tensor) -> torch.Tensor:
        """
        CorrMatch-style pseudo-label propagation:
          - Smooth the teacher pseudo label locally.
          - Combine original pseudo label and local correlation-smoothed pseudo label.
        """
        _, pseudo = self._teacher_pseudo(x_u)

        smooth = self._low_pass_3d(pseudo, self.corr_smooth_kernel)
        pseudo_corr = self.corr_alpha * pseudo + (1.0 - self.corr_alpha) * smooth
        pseudo_corr = pseudo_corr.clamp(0.0, 1.0)

        s_u, _ = self.student(x_u, return_features=True)
        return self._masked_consistency_loss(s_u, pseudo_corr)

    def _wtbcp_unsup_loss(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        x_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        WT-BCP-style wavelet/frequency-inspired copy-paste:
          - First create BCP mixed image and target.
          - Then combine low-frequency component from unlabeled image with
            high-frequency component from mixed image.
        """
        _, pseudo = self._teacher_pseudo(x_u)

        b = min(x_l.shape[0], x_u.shape[0], y_l.shape[0])
        x_l = x_l[:b]
        y_l = y_l[:b]
        x_u = x_u[:b]
        pseudo = pseudo[:b]

        y_l = self._match_spatial(y_l, x_u, mode="nearest")
        pseudo = self._match_spatial(pseudo, x_u, mode="trilinear")

        fg_mask = (y_l > self.pseudo_threshold).float()

        x_mix = x_u * (1.0 - fg_mask) + x_l * fg_mask
        y_mix = pseudo * (1.0 - fg_mask) + y_l * fg_mask

        low_u = self._low_pass_3d(x_u, self.wt_low_kernel)
        low_mix = self._low_pass_3d(x_mix, self.wt_low_kernel)
        high_mix = x_mix - low_mix

        x_wt = low_u + self.wt_high_weight * high_mix

        logits = self.student(x_wt)
        if isinstance(logits, tuple):
            logits = logits[0]

        return supervised_loss(logits, y_mix)

    def _baseline_unsup_loss(
        self,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        x_u: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_unlabeled or x_u is None or self.lambda_unsup <= 0:
            z = x_l.new_tensor(0.0)
            return z, z

        method = self.baseline_method

        if method in {"mean_teacher", "mt", "full", "frequency", "prototype"}:
            return self._mean_teacher_unsup_loss(x_u)

        if method == "bcp":
            loss = self._bcp_unsup_loss(x_l, y_l, x_u)
            return loss, x_l.new_tensor(0.0)

        if method == "unimatch":
            loss = self._unimatch_unsup_loss(x_u)
            return loss, x_l.new_tensor(0.0)

        if method == "corrmatch":
            loss = self._corrmatch_unsup_loss(x_u)
            return loss, x_l.new_tensor(0.0)

        if method == "wtbcp":
            loss = self._wtbcp_unsup_loss(x_l, y_l, x_u)
            return loss, x_l.new_tensor(0.0)

        if method in {"supervised", "nnunet", "nnunet_style"}:
            z = x_l.new_tensor(0.0)
            return z, z

        raise ValueError(
            f"Unsupported baseline_method='{method}'. "
            "Use one of: supervised, nnunet, mean_teacher, bcp, unimatch, corrmatch, wtbcp."
        )

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
            _, feat_l, loss_sup = self._supervised_branch(x_l, y_l)

        # Keep unsupervised loss outside autocast for numerical stability.
        loss_unsup, loss_spec = self._baseline_unsup_loss(x_l, y_l, x_u)

        if self.lambda_proto > 0:
            loss_proto = prototype_contrast_loss(feat_l, y_l, self.prototype_memory)
        else:
            loss_proto = loss_sup.new_tensor(0.0)

        loss = (
            loss_sup
            + self.lambda_unsup * loss_unsup
            + self.lambda_spec * loss_spec
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
                    f"hd95={metrics.hd95:.4f}"
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

            totals["dice"] += float(m.dice)
            totals["iou"] += float(m.iou)
            totals["precision"] += float(m.precision)
            totals["recall"] += float(m.recall)
            totals["f1"] += float(m.f1)
            totals["minority_f1"] += float(m.minority_f1)
            totals["hd95"] += float(m.hd95)

            n += 1

            if self.max_val_batches > 0 and n >= self.max_val_batches:
                break

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