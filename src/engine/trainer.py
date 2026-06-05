from __future__ import annotations

import csv
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
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


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix


def _safe_scalar(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0).item())


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _estimate_class_weights(loader, device: torch.device) -> torch.Tensor:
    pos = None
    total = None
    for batch in loader:
        y = batch["label"].to(device).float()
        # sum over B and spatial dims, keep C
        dims = (0,) + tuple(range(2, y.ndim))
        cur_pos = y.sum(dim=dims)  # [C]
        vox_per_sample = y[0].numel() // y.shape[1]
        cur_total = torch.tensor(vox_per_sample, device=device).repeat(y.shape[1]) * y.shape[0]
        pos = cur_pos if pos is None else pos + cur_pos
        total = cur_total if total is None else total + cur_total

    freq = (pos / total.clamp_min(1.0)).clamp(min=1e-6, max=1 - 1e-6)
    # inverse-frequency like weighting
    w = 1.0 / torch.log(freq + 1.02)
    return torch.nan_to_num(w, nan=1.0, posinf=3.0, neginf=1.0).clamp(0.5, 5.0).detach()


class MeanTeacherTrainer:
    def __init__(self, student: torch.nn.Module, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = student.to(self.device)
        self.teacher = deepcopy(student).to(self.device)
        self.teacher.eval()

        # reliability / gate heads
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
        self.optim = Adam(params, lr=float(cfg["train"]["lr"]))
        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=int(cfg["train"]["epochs"]),
            eta_min=float(cfg["train"].get("min_lr", 1e-6)),
        )

        self.epochs = int(cfg["train"]["epochs"])
        self.grad_clip = float(cfg["train"].get("grad_clip", 1.0))
        self.ema_m = float(cfg["train"].get("ema_momentum", 0.99))
        self.mixup_alpha = float(cfg["train"].get("mixup_alpha", 0.0))
        self.warmup_epochs = int(cfg["train"].get("warmup_epochs", 10))

        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # loss weights
        self.l_ssl = float(cfg["loss"].get("lambda_ssl", 1.0))
        self.l_minor = float(cfg["loss"].get("lambda_minor", 0.8))
        self.l_struct = float(cfg["loss"].get("lambda_struct", 0.1))
        self.l_feat = float(cfg["loss"].get("lambda_feat_consistency", 0.0))
        self.base_tau = float(cfg["loss"].get("tau", 0.65))

    def _tau_schedule(self, epoch: int) -> float:
        # curriculum-like schedule:
        # early strict -> middle base -> late slightly relaxed
        if epoch <= self.warmup_epochs:
            return min(0.80, self.base_tau + 0.10)
        if epoch <= 2 * self.warmup_epochs:
            return self.base_tau
        return max(0.55, self.base_tau - 0.05)

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

                # save best checkpoint only on finite/valid dice
                if torch.isfinite(torch.tensor(metrics.dice)) and metrics.dice > best_dice:
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

        tau = self._tau_schedule(epoch)

        rel_cfg = dict(self.cfg["loss"].get("reliability", {}))
        ab = self.cfg.get("ablation_switches", {})

        if "minority_score" in ab:
            rel_cfg["enable_minority_score"] = bool(ab["minority_score"])
        if "ood" in ab:
            rel_cfg["enable_ood"] = bool(ab["ood"])
        if "reliability" in ab:
            rel_cfg["enable_reliability"] = bool(ab["reliability"])

        # curriculum gates
        enable_reliability = bool(rel_cfg.get("enable_reliability", True)) and epoch > self.warmup_epochs
        enable_consistency = bool(ab.get("consistency", True)) and epoch > self.warmup_epochs
        struct_lambda = self.l_struct if epoch > 2 * self.warmup_epochs else 0.0
        feat_lambda = self.l_feat if enable_consistency else 0.0
        ssl_lambda = self.l_ssl * (0.3 if epoch <= self.warmup_epochs else 1.0)

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
            class_weights = torch.tensor(
                self.cfg["loss"].get("minor_class_weights", [1.0]),
                device=self.device,
                dtype=torch.float32,
            )

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
                # supervised
                s_l = self.student(x_l)
                l_sup = supervised_loss(s_l, y_l)

                # teacher / student on unlabeled
                with torch.no_grad():
                    t_out = self.teacher(x_u)
                    if isinstance(t_out, tuple):
                        t_u, t_feat = t_out
                    else:
                        t_u, t_feat = t_out, None
                    t_u = torch.sigmoid(t_u).clamp(0.0, 1.0)

                s_out = self.student(x_u)
                if isinstance(s_out, tuple):
                    s_u, s_feat = s_out
                else:
                    s_u, s_feat = s_out, None
                s_u_prob = torch.sigmoid(s_u).clamp(0.0, 1.0)

                # reliability maps
                rel_parts = reliability_components(
                    s_u_prob.detach(),
                    t_u,
                    x_u,
                    student_feat=s_feat.detach() if s_feat is not None else None,
                    teacher_feat=t_feat.detach() if t_feat is not None else None,
                    temporal_teacher_probs=None,
                    bank_mean=None,
                    bank_var=None,
                    enable_ood=bool(rel_cfg.get("enable_ood", True)),
                    enable_consistency=enable_consistency,
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
                stacked = torch.cat(maps_for_cat, dim=1)
                reliability = torch.sigmoid(self.reliability_mlp(stacked)).clamp(0.0, 1.0)

                if minority_score_u is not None and bool(rel_cfg.get("enable_minority_score", True)):
                    if not torch.is_tensor(minority_score_u):
                        minority_score_u = torch.as_tensor(minority_score_u, device=self.device)
                    minority_score_u = minority_score_u.to(self.device).float()
                    boost = minority_score_u.view(-1, 1, *([1] * (reliability.ndim - 2)))
                    reliability = reliability * (1.0 + float(rel_cfg.get("minority_reliability_boost", 0.15)) * boost)
                    reliability = reliability.clamp(0.0, 1.0)

                pseudo_w, _ = dynamic_pseudo_weight(t_u, tau=tau)
                if enable_reliability:
                    fused_weight = fuse_pseudo_with_reliability(
                        pseudo_w,
                        reliability,
                        gate_mlp=self.fusion_gate_mlp,
                        mode=rel_cfg.get("fusion_mode", "convex"),
                        alpha=float(rel_cfg.get("fusion_alpha", 0.75)),
                    )
                else:
                    fused_weight = pseudo_w

                l_unsup = unsupervised_loss(s_u, t_u, tau=tau, fused_weight=fused_weight)

                if enable_consistency and (s_feat is not None) and (t_feat is not None):
                    l_feat = feature_consistency_loss(s_feat, t_feat)
                else:
                    l_feat = torch.zeros((), device=self.device)

                l_minor = minority_sensitive_loss(
                    s_l,
                    y_l,
                    class_weights=class_weights,
                    focal_alpha=float(self.cfg["loss"].get("focal_alpha", 0.25)),
                    focal_gamma=float(self.cfg["loss"].get("focal_gamma", 2.0)),
                )

                if struct_lambda > 0:
                    l_struct = structural_loss(
                        s_l,
                        y_l,
                        hd_weight=float(self.cfg["loss"].get("hd_weight", 0.25)),
                        fg_weight=float(self.cfg["loss"].get("fg_weight", 2.0)),
                        topo_weight=float(self.cfg["loss"].get("topo_weight", 0.08)),
                    )
                else:
                    l_struct = torch.zeros((), device=self.device)

                loss = l_sup + ssl_lambda * l_unsup + self.l_minor * l_minor + struct_lambda * l_struct + feat_lambda * l_feat
                loss = loss / self.grad_accum_steps

            # ===== non-finite guard =====
            if not _is_finite_tensor(loss):
                print(f"[WARN] non-finite loss at epoch={epoch}, step={n_steps + 1}; skip this step.")
                self.optim.zero_grad(set_to_none=True)
                n_steps += 1
                continue

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

            running += _safe_scalar(loss * self.grad_accum_steps)
            conf_running += _safe_scalar(pseudo_w.mean())
            rel_running += _safe_scalar(reliability.mean())
            ood_running += _safe_scalar(rel_parts["ood_map"].mean())
            cons_running += _safe_scalar(rel_parts["consistency_map"].mean())
            n_steps += 1

        # flush remainder grads
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

            m = compute_binary_metrics(logits, y, threshold=float(self.cfg["inference"].get("threshold", 0.4)))

            # extra clamp for safety
            agg["dice"] += max(0.0, min(1.0, float(m.dice)))
            agg["iou"] += max(0.0, min(1.0, float(m.iou)))
            agg["precision"] += max(0.0, min(1.0, float(m.precision)))
            agg["recall"] += max(0.0, min(1.0, float(m.recall)))
            agg["f1"] += max(0.0, min(1.0, float(m.f1)))
            agg["minority_f1"] += max(0.0, min(1.0, float(m.minority_f1)))

            hd = float(m.hd95)
            if not torch.isfinite(torch.tensor(hd)):
                hd = 1e3
            agg["hd95"] += hd
            n += 1

        from utils.metrics import SegMetrics
        d = max(n, 1)
        return SegMetrics(
            agg["dice"] / d,
            agg["iou"] / d,
            agg["precision"] / d,
            agg["recall"] / d,
            agg["f1"] / d,
            agg["minority_f1"] / d,
            agg["hd95"] / d,
        )