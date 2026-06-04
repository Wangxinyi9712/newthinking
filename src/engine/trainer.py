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
    dynamic_pseudo_weight,
    feature_consistency_loss,
    fuse_pseudo_with_reliability,
    minority_sensitive_loss,
    reliability_components,
    structural_loss,
    supervised_loss,
    unsupervised_loss,
)
from utils.metrics import compute_binary_metrics


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix


def _estimate_class_weights(loader, device: torch.device) -> torch.Tensor:
    pos = None
    total = None
    for batch in loader:
        y = batch["label"].to(device).float()
        dims = tuple(range(0, y.ndim))[0:1] + tuple(range(2, y.ndim))
        cur_pos = y.sum(dim=dims)
        cur_total = torch.tensor(y[0].numel() / y.shape[1], device=device).repeat(y.shape[1]) * y.shape[0]
        pos = cur_pos if pos is None else pos + cur_pos
        total = cur_total if total is None else total + cur_total
    freq = (pos / total.clamp_min(1.0)).clamp(min=1e-6, max=1 - 1e-6)
    return (1.0 / torch.log(freq + 1.02)).detach()


def _resize_to_spatial(x: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    if x.shape[2:] == spatial:
        return x
    if x.ndim == 5:
        return F.interpolate(x, size=spatial, mode="trilinear", align_corners=False)
    if x.ndim == 4:
        return F.interpolate(x, size=spatial, mode="bilinear", align_corners=False)
    return x


def _align_maps_for_cat(maps: list[torch.Tensor], ref: torch.Tensor | None = None) -> list[torch.Tensor]:
    if len(maps) == 0:
        return maps
    ref_tensor = ref if ref is not None else maps[0]
    ref_spatial = tuple(ref_tensor.shape[2:])
    return [_resize_to_spatial(m, ref_spatial) for m in maps]


class MeanTeacherTrainer:
    def __init__(self, student: torch.nn.Module, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        dim = 3 if cfg["data"].get("dim", "3d") == "3d" else 2
        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        self.reliability_mlp = torch.nn.Sequential(
            conv(9, 16, kernel_size=1),
            torch.nn.GELU(),
            conv(16, 1, kernel_size=1),
        ).to(self.device)

        self.fusion_gate_mlp = torch.nn.Sequential(
            conv(2, 8, kernel_size=1),
            torch.nn.GELU(),
            conv(8, 1, kernel_size=1),
        ).to(self.device)

        params = (
            list(self.student.parameters())
            + list(self.reliability_mlp.parameters())
            + list(self.fusion_gate_mlp.parameters())
        )
        self.optim = Adam(params, lr=cfg["train"]["lr"])
        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("min_lr", 1e-6),
        )

        self.epochs = cfg["train"]["epochs"]
        self.grad_clip = float(cfg["train"].get("grad_clip", 0.0))
        self.ema_m = float(cfg["train"].get("ema_momentum", 0.99))
        self.mixup_alpha = float(cfg["train"].get("mixup_alpha", 0.0))
        self.warmup_epochs = int(cfg["train"].get("warmup_epochs", 10))

        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.l_ssl = float(cfg["loss"].get("lambda_ssl", 1.0))
        self.l_minor = float(cfg["loss"].get("lambda_minor", 1.0))
        self.l_struct = float(cfg["loss"].get("lambda_struct", 0.2))
        self.l_feat = float(cfg["loss"].get("lambda_feat_consistency", 0.1))
        self.tau = float(cfg["loss"].get("tau", 0.7))

        self.temporal_momentum = float(cfg["loss"].get("temporal_momentum", 0.9))
        self.feature_bank_momentum = float(cfg["loss"].get("feature_bank_momentum", 0.95))
        self.temporal_teacher_probs: torch.Tensor | None = None
        self.feature_bank_mean: torch.Tensor | None = None
        self.feature_bank_var: torch.Tensor | None = None

    def _update_feature_bank(self, teacher_feat: torch.Tensor) -> None:
        reduce_dims = (0,) + tuple(range(2, teacher_feat.ndim))
        cur_mean = teacher_feat.detach().mean(dim=reduce_dims, keepdim=True)
        cur_var = teacher_feat.detach().var(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.feature_bank_mean is None or self.feature_bank_var is None:
            self.feature_bank_mean = cur_mean
            self.feature_bank_var = cur_var
            return
        m = self.feature_bank_momentum
        self.feature_bank_mean = m * self.feature_bank_mean + (1.0 - m) * cur_mean
        self.feature_bank_var = m * self.feature_bank_var + (1.0 - m) * cur_var

    def fit(self, loaders: dict, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        best_dice = -1.0
        history_file = out / "history.csv"

        with open(history_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "unsup_conf_mean",
                    "reliability_mean",
                    "ood_mean",
                    "consistency_mean",
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
                train_loss, conf_mean, rel_mean, ood_mean, cons_mean = self._train_one_epoch(loaders, epoch=epoch)
                self.scheduler.step()

                metrics = self.evaluate(
                    loaders["val"],
                    use_teacher_ema=bool(self.cfg["inference"].get("use_teacher_ema", True)),
                )
                lr = self.optim.param_groups[0]["lr"]

                writer.writerow(
                    [
                        epoch,
                        lr,
                        train_loss,
                        conf_mean,
                        rel_mean,
                        ood_mean,
                        cons_mean,
                        metrics.dice,
                        metrics.iou,
                        metrics.precision,
                        metrics.recall,
                        metrics.f1,
                        metrics.minority_f1,
                        metrics.hd95,
                    ]
                )

                if metrics.dice > best_dice:
                    best_dice = metrics.dice
                    torch.save(
                        {
                            "student": self.student.state_dict(),
                            "teacher": self.teacher.state_dict(),
                            "reliability_mlp": self.reliability_mlp.state_dict(),
                            "fusion_gate_mlp": self.fusion_gate_mlp.state_dict(),
                        },
                        out / "best.pt",
                    )

    def _train_one_epoch(self, loaders: dict, epoch: int) -> tuple[float, float, float, float, float]:
        self.student.train()
        self.reliability_mlp.train()
        self.fusion_gate_mlp.train()

        unlabeled_iter = iter(loaders["unlabeled"])
        running = 0.0
        conf_running = 0.0
        rel_running = 0.0
        ood_running = 0.0
        cons_running = 0.0
        n_steps = 0

        if self.cfg["loss"].get("auto_class_weights", False):
            class_weights = _estimate_class_weights(loaders["labeled"], self.device)
        else:
            class_weights = torch.tensor(self.cfg["loss"].get("minor_class_weights", [1.0]), device=self.device)

        self.optim.zero_grad(set_to_none=True)

        pbar = tqdm(
            loaders["labeled"],
            desc=f"train-{epoch}/{self.epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in pbar:
            try:
                ub = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(loaders["unlabeled"])
                ub = next(unlabeled_iter)

            x_l = batch["image"].to(self.device)
            y_l = batch["label"].to(self.device)
            x_u = ub["image"].to(self.device)
            minority_score_u = ub.get("minority_score", None)

            x_l, y_l = _mixup(x_l, y_l, alpha=self.mixup_alpha)

            amp_ctx = (lambda: torch.amp.autocast("cuda")) if self.use_amp else nullcontext
            with amp_ctx():
                s_l = self.student(x_l)
                l_sup = supervised_loss(s_l, y_l)

                with torch.no_grad():
                    t_u, t_feat = self.teacher(x_u, return_features=True)
                    t_u = torch.sigmoid(t_u)
                    self._update_feature_bank(t_feat)

                s_u, s_feat = self.student(x_u, return_features=True)
                s_u_prob = torch.sigmoid(s_u)

                rel_cfg = dict(self.cfg["loss"].get("reliability", {}))
                ab = self.cfg.get("ablation_switches", {})
                if "minority_score" in ab:
                    rel_cfg["enable_minority_score"] = bool(ab["minority_score"])
                if "ood" in ab:
                    rel_cfg["enable_ood"] = bool(ab["ood"])
                if "reliability" in ab:
                    rel_cfg["enable_reliability"] = bool(ab["reliability"])
                use_consistency = bool(ab.get("consistency", True))

                # warmup：前几轮关闭易不稳定分支，先学稳主干
                if epoch <= self.warmup_epochs:
                    rel_cfg["enable_reliability"] = False
                    use_consistency = False
                    struct_lambda = 0.0
                else:
                    struct_lambda = self.l_struct

                if minority_score_u is not None:
                    if not torch.is_tensor(minority_score_u):
                        minority_score_u = torch.as_tensor(minority_score_u, device=self.device)
                    minority_score_u = minority_score_u.to(self.device).float()

                rel_parts = reliability_components(
                    s_u_prob.detach(),
                    t_u,
                    x_u,
                    student_feat=s_feat.detach(),
                    teacher_feat=t_feat,
                    temporal_teacher_probs=(
                        self.temporal_teacher_probs
                        if self.temporal_teacher_probs is not None and self.temporal_teacher_probs.shape == t_u.shape
                        else None
                    ),
                    bank_mean=self.feature_bank_mean,
                    bank_var=self.feature_bank_var,
                    enable_ood=bool(rel_cfg.get("enable_ood", True)),
                    enable_consistency=use_consistency,
                )

                maps_for_cat = [
                    rel_parts["confidence_map"],
                    rel_parts["entropy_map"],
                    rel_parts["consistency_map"],
                    rel_parts["ood_map"],
                    rel_parts["feature_distance_map"],
                    rel_parts["feature_embedding_map"],
                    rel_parts["gradient_uncertainty_map"],
                    rel_parts["temporal_consistency_map"],
                    rel_parts["transformer_feature_map"],
                ]
                maps_for_cat = _align_maps_for_cat(maps_for_cat, ref=rel_parts["confidence_map"])
                stacked = torch.cat(maps_for_cat, dim=1)
                reliability = torch.sigmoid(self.reliability_mlp(stacked))

                if minority_score_u is not None and bool(rel_cfg.get("enable_minority_score", True)):
                    boost = minority_score_u.view(-1, 1, *([1] * (reliability.ndim - 2))).to(reliability.device)
                    reliability = reliability * (1.0 + float(rel_cfg.get("minority_reliability_boost", 0.3)) * boost)
                reliability = reliability.clamp(0, 1)

                pseudo_w, _ = dynamic_pseudo_weight(t_u, self.tau)
                if rel_cfg.get("enable_reliability", True):
                    fused_weight = fuse_pseudo_with_reliability(
                        pseudo_w,
                        reliability,
                        gate_mlp=self.fusion_gate_mlp,
                        mode=rel_cfg.get("fusion_mode", "learnable"),
                        alpha=float(rel_cfg.get("fusion_alpha", 0.5)),
                    )
                else:
                    fused_weight = pseudo_w

                l_unsup = unsupervised_loss(s_u, t_u, self.tau, fused_weight=fused_weight)
                l_feat = feature_consistency_loss(s_feat, t_feat) if use_consistency else torch.zeros((), device=self.device)

                l_minor = minority_sensitive_loss(
                    s_l,
                    y_l,
                    class_weights=class_weights,
                    focal_alpha=float(self.cfg["loss"].get("focal_alpha", 0.25)),
                    focal_gamma=float(self.cfg["loss"].get("focal_gamma", 2.0)),
                )

                l_struct = structural_loss(
                    s_l,
                    y_l,
                    hd_weight=float(self.cfg["loss"].get("hd_weight", 0.5)),
                    fg_weight=float(self.cfg["loss"].get("fg_weight", 2.0)),
                    topo_weight=float(self.cfg["loss"].get("topo_weight", 0.2)),
                )

                l_feat_lambda = self.l_feat if use_consistency else 0.0
                loss = l_sup + self.l_ssl * l_unsup + self.l_minor * l_minor + struct_lambda * l_struct + l_feat_lambda * l_feat
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            step_now = (n_steps + 1) % self.grad_accum_steps == 0
            if step_now:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.student.parameters())
                        + list(self.reliability_mlp.parameters())
                        + list(self.fusion_gate_mlp.parameters()),
                        self.grad_clip,
                    )
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                update_ema(self.student, self.teacher, self.ema_m)
                if self.temporal_teacher_probs is None:
                    self.temporal_teacher_probs = t_u.detach()
                else:
                    m = self.temporal_momentum
                    self.temporal_teacher_probs = m * self.temporal_teacher_probs + (1.0 - m) * t_u.detach()

            running += float(loss.item() * self.grad_accum_steps)
            conf_running += float(pseudo_w.mean().item())
            rel_running += float(reliability.mean().item())
            ood_running += float(rel_parts["ood_map"].mean().item())
            cons_running += float(rel_parts["consistency_map"].mean().item())
            n_steps += 1

        if n_steps % self.grad_accum_steps != 0:
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters())
                    + list(self.reliability_mlp.parameters())
                    + list(self.fusion_gate_mlp.parameters()),
                    self.grad_clip,
                )
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)

            update_ema(self.student, self.teacher, self.ema_m)
            if self.temporal_teacher_probs is None:
                self.temporal_teacher_probs = t_u.detach()
            else:
                m = self.temporal_momentum
                self.temporal_teacher_probs = m * self.temporal_teacher_probs + (1.0 - m) * t_u.detach()

        d = max(n_steps, 1)
        return running / d, conf_running / d, rel_running / d, ood_running / d, cons_running / d

    @torch.no_grad()
    def evaluate(self, loader, use_teacher_ema: bool = True, use_ensemble: bool = False) -> object:
        self.student.eval()
        self.teacher.eval()

        agg = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "minority_f1": 0.0,
            "hd95": 0.0,
        }
        n = 0

        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            amp_ctx = (lambda: torch.amp.autocast("cuda")) if self.use_amp else nullcontext
            with amp_ctx():
                if use_ensemble:
                    logits = 0.5 * self.student(x) + 0.5 * self.teacher(x)
                else:
                    logits = self.teacher(x) if use_teacher_ema else self.student(x)

            m = compute_binary_metrics(logits, y, threshold=float(self.cfg["inference"].get("threshold", 0.5)))
            agg["dice"] += m.dice
            agg["iou"] += m.iou
            agg["precision"] += m.precision
            agg["recall"] += m.recall
            agg["f1"] += m.f1
            agg["minority_f1"] += m.minority_f1
            agg["hd95"] += m.hd95
            n += 1

        from utils.metrics import SegMetrics
        return SegMetrics(*(agg[k] / max(n, 1) for k in ["dice", "iou", "precision", "recall", "f1", "minority_f1", "hd95"]))