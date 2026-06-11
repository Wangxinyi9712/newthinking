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
from models.discriminator import SegDiscriminator
from utils.metrics import compute_binary_metrics


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))


def _safe_scalar(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0).item())


def _is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


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

        self.reliability_mlp = torch.nn.Sequential(conv(9, 16, 1), torch.nn.GELU(), conv(16, 1, 1)).to(self.device)
        self.fusion_gate_mlp = torch.nn.Sequential(conv(2, 8, 1), torch.nn.GELU(), conv(8, 1, 1)).to(self.device)

        self.use_adv = bool(cfg["loss"].get("use_adversarial", True))
        self.lambda_adv = float(cfg["loss"].get("lambda_adv", 0.05))
        self.lambda_d = float(cfg["loss"].get("lambda_d", 0.5))
        self.d_steps = int(cfg["loss"].get("d_steps", 1))

        if self.use_adv:
            self.discriminator = SegDiscriminator(
                in_channels=int(cfg["model"]["in_channels"]),
                out_channels=int(cfg["model"]["out_channels"]),
                base_ch=32,
                dim=dim,
            ).to(self.device)
        else:
            self.discriminator = None

        g_params = (
            list(self.student.parameters())
            + list(self.student_aux.parameters())
            + list(self.reliability_mlp.parameters())
            + list(self.fusion_gate_mlp.parameters())
        )
        self.optim_g = Adam(g_params, lr=float(cfg["train"]["lr"]))

        if self.use_adv:
            self.optim_d = Adam(self.discriminator.parameters(), lr=float(cfg["train"]["lr"]) * 0.5)
        else:
            self.optim_d = None

        self.scheduler = CosineAnnealingLR(
            self.optim_g, T_max=int(cfg["train"]["epochs"]), eta_min=float(cfg["train"].get("min_lr", 1e-6))
        )

        self.epochs = int(cfg["train"]["epochs"])
        self.grad_clip = float(cfg["train"].get("grad_clip", 1.0))
        self.ema_m = float(cfg["train"].get("ema_momentum", 0.99))
        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.warmup_epochs = int(cfg["train"].get("warmup_epochs", 12))
        self.base_tau = float(cfg["loss"].get("tau", 0.65))
        self.lambda_ssl = float(cfg["loss"].get("lambda_ssl", 1.0))
        self.lambda_minor = float(cfg["loss"].get("lambda_minor", 0.8))
        self.lambda_struct = float(cfg["loss"].get("lambda_struct", 0.08))
        self.lambda_feat = float(cfg["loss"].get("lambda_feat_consistency", 0.0))
        self.lambda_cps = float(cfg["loss"].get("lambda_cps", 0.4))
        self.lambda_sdm = float(cfg["loss"].get("lambda_sdm", 0.2))

        self.teacher_temp = float(cfg["loss"].get("teacher_temperature", 1.5))
        self.tau_quantile = float(cfg["loss"].get("tau_quantile", 0.70))
        self.soft_gate_power = float(cfg["loss"].get("soft_gate_power", 1.0))
        self.cvar_ratio = float(cfg["loss"].get("cvar_ratio", 0.20))

    def _curriculum(self, epoch: int):
        if epoch <= self.warmup_epochs:
            return 0.3, False, False, 0.0
        if epoch <= 2 * self.warmup_epochs:
            return 1.0, True, True, 0.0
        return 1.0, True, True, self.lambda_struct

    def _build_ckpt_dict(self) -> dict:
        ckpt = {
            "student": self.student.state_dict(),
            "student_aux": self.student_aux.state_dict(),
            "teacher": self.teacher.state_dict(),
            "reliability_mlp": self.reliability_mlp.state_dict(),
            "fusion_gate_mlp": self.fusion_gate_mlp.state_dict(),
        }
        if self.use_adv and self.discriminator is not None:
            ckpt["discriminator"] = self.discriminator.state_dict()
        return ckpt

    def fit(self, loaders: dict, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        best_dice = -1.0
        best_epoch = -1
        history_path = out / "history.csv"

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch", "lr", "train_loss",
                    "unsup_conf_mean", "reliability_mean", "ood_mean", "consistency_mean",
                    "val_dice", "val_iou", "val_precision", "val_recall", "val_f1", "val_minority_f1", "val_hd95",
                ]
            )

            for epoch in range(1, self.epochs + 1):
                tl, cm, rm, om, sm = self._train_one_epoch(loaders, epoch)
                self.scheduler.step()

                m = self.evaluate(loaders["val"], use_teacher_ema=bool(self.cfg["inference"].get("use_teacher_ema", True)))
                lr = self.optim_g.param_groups[0]["lr"]

                w.writerow([epoch, lr, tl, cm, rm, om, sm, m.dice, m.iou, m.precision, m.recall, m.f1, m.minority_f1, m.hd95])

                if torch.isfinite(torch.tensor(m.dice)) and m.dice > best_dice:
                    best_dice = float(m.dice)
                    best_epoch = epoch
                    torch.save(self._build_ckpt_dict(), out / "best.pt")

        # 始终保存最后一轮模型
        torch.save(self._build_ckpt_dict(), out / "last.pt")

        # 可选：记录摘要
        with open(out / "checkpoint_summary.txt", "w", encoding="utf-8") as fp:
            fp.write(f"best_epoch: {best_epoch}\n")
            fp.write(f"best_dice: {best_dice:.6f}\n")
            fp.write("best_ckpt: best.pt\n")
            fp.write("last_ckpt: last.pt\n")

    def _train_one_epoch(self, loaders: dict, epoch: int):
        self.student.train()
        self.student_aux.train()
        self.reliability_mlp.train()
        self.fusion_gate_mlp.train()
        if self.use_adv:
            self.discriminator.train()

        ssl_factor, use_rel, use_cons, struct_lambda = self._curriculum(epoch)
        ssl_lambda = self.lambda_ssl * ssl_factor

        rel_cfg = dict(self.cfg["loss"].get("reliability", {}))
        ab = self.cfg.get("ablation_switches", {})
        if "ood" in ab:
            rel_cfg["enable_ood"] = bool(ab["ood"])
        if "reliability" in ab:
            use_rel = use_rel and bool(ab["reliability"])
        if "consistency" in ab:
            use_cons = use_cons and bool(ab["consistency"])

        unlabeled_iter = iter(loaders["unlabeled"])
        running = conf_running = rel_running = ood_running = cons_running = 0.0
        n_steps = 0

        class_weights = torch.tensor(self.cfg["loss"].get("minor_class_weights", [2.0]), device=self.device, dtype=torch.float32)
        self.optim_g.zero_grad(set_to_none=True)
        if self.use_adv:
            self.optim_d.zero_grad(set_to_none=True)

        pbar = tqdm(loaders["labeled"], desc=f"train-{epoch}/{self.epochs}", leave=True, dynamic_ncols=True)
        for batch in pbar:
            try:
                ub = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(loaders["unlabeled"])
                ub = next(unlabeled_iter)

            x_l = batch["image"].to(self.device)
            y_l = batch["label"].to(self.device)
            x_u = ub["image"].to(self.device)

            amp_ctx = (lambda: torch.amp.autocast("cuda")) if self.use_amp else nullcontext
            with amp_ctx():
                s_l_logits, s_l_sdm, s_l_feat = self.student(x_l, return_features=True)
                s2_l_logits, s2_l_sdm, _ = self.student_aux(x_l, return_features=True)

                l_sup = supervised_loss(s_l_logits, y_l)
                l_sup_aux = supervised_loss(s2_l_logits, y_l)
                l_sdm = sdm_loss(s_l_sdm, y_l) + sdm_loss(s2_l_sdm, y_l)

                with torch.no_grad():
                    t_logits, t_sdm, t_feat = self.teacher(x_u, return_features=True)
                    t_u = teacher_prob_with_temperature(t_logits, self.teacher_temp).float()

                s_u_logits, s_u_sdm, s_u_feat = self.student(x_u, return_features=True)
                s2_u_logits, s2_u_sdm, s2_u_feat = self.student_aux(x_u, return_features=True)
                s_u_prob = torch.sigmoid(s_u_logits.float()).clamp(0.0, 1.0)

                tau_q = adaptive_tau_from_quantile(t_u, q=self.tau_quantile, min_tau=0.50, max_tau=0.90)
                tau = 0.5 * self.base_tau + 0.5 * tau_q

                rel_parts = reliability_components(
                    s_u_prob.detach(), t_u, x_u.float(),
                    enable_ood=bool(rel_cfg.get("enable_ood", True)),
                    enable_consistency=use_cons,
                )

                stacked = torch.cat(
                    [
                        rel_parts["confidence_map"], rel_parts["entropy_map"], rel_parts["consistency_map"], rel_parts["ood_map"],
                        rel_parts["feature_distance_map"], rel_parts["feature_embedding_map"], rel_parts["gradient_uncertainty_map"],
                        rel_parts["temporal_consistency_map"], rel_parts["transformer_feature_map"],
                    ],
                    dim=1,
                ).float()

                reliability = torch.sigmoid(self.reliability_mlp(stacked)).clamp(0.0, 1.0)
                pseudo_w, conf_mask = dynamic_pseudo_weight(t_u, tau=tau)

                fused = (
                    fuse_pseudo_with_reliability(
                        pseudo_w, reliability,
                        gate_mlp=self.fusion_gate_mlp if use_rel else None,
                        mode=rel_cfg.get("fusion_mode", "convex"),
                        alpha=float(rel_cfg.get("fusion_alpha", 0.75)),
                    )
                    if use_rel
                    else pseudo_w
                )
                soft_gate = reliability.pow(self.soft_gate_power)

                l_unsup = unsupervised_loss(s_u_logits.float(), t_u, tau=tau, fused_weight=fused, soft_gate=soft_gate, cvar_ratio=self.cvar_ratio)
                l_unsup_aux = unsupervised_loss(s2_u_logits.float(), t_u, tau=tau, fused_weight=fused, soft_gate=soft_gate, cvar_ratio=self.cvar_ratio)
                l_cps = cps_loss(s_u_logits.float(), s2_u_logits.float(), conf_mask=conf_mask)

                l_minor = minority_sensitive_loss(
                    s_l_logits.float(), y_l.float(), class_weights,
                    focal_alpha=float(self.cfg["loss"].get("focal_alpha", 0.25)),
                    focal_gamma=float(self.cfg["loss"].get("focal_gamma", 2.0)),
                )
                l_struct = (
                    structural_loss(
                        s_l_logits.float(), y_l.float(),
                        hd_weight=float(self.cfg["loss"].get("hd_weight", 0.25)),
                        fg_weight=float(self.cfg["loss"].get("fg_weight", 2.0)),
                        topo_weight=float(self.cfg["loss"].get("topo_weight", 0.08)),
                        smooth_edge=True,
                    )
                    if struct_lambda > 0 else torch.zeros((), device=self.device)
                )
                l_feat = feature_consistency_loss(s_u_feat.float(), t_feat.float()) if use_cons else torch.zeros((), device=self.device)

                l_adv_g = torch.zeros((), device=self.device)
                if self.use_adv:
                    p_fake = torch.sigmoid(s_u_logits.float())
                    d_fake = self.discriminator(x_u.float(), p_fake)
                    l_adv_g = gan_generator_loss(d_fake)

                g_loss = (
                    l_sup + l_sup_aux
                    + ssl_lambda * (l_unsup + l_unsup_aux)
                    + self.lambda_cps * l_cps
                    + self.lambda_minor * l_minor
                    + struct_lambda * l_struct
                    + self.lambda_feat * l_feat
                    + self.lambda_sdm * l_sdm
                    + self.lambda_adv * l_adv_g
                ) / self.grad_accum_steps

            if not _is_finite(g_loss):
                self.optim_g.zero_grad(set_to_none=True)
                if self.use_adv:
                    self.optim_d.zero_grad(set_to_none=True)
                n_steps += 1
                continue

            self.scaler_g.scale(g_loss).backward()

            if (n_steps + 1) % self.grad_accum_steps == 0:
                self.scaler_g.unscale_(self.optim_g)
                torch.nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.student_aux.parameters())
                    + list(self.reliability_mlp.parameters()) + list(self.fusion_gate_mlp.parameters()),
                    self.grad_clip,
                )
                self.scaler_g.step(self.optim_g)
                self.scaler_g.update()
                self.optim_g.zero_grad(set_to_none=True)

                update_ema(self.student, self.teacher, self.ema_m)

                if self.use_adv:
                    for _ in range(self.d_steps):
                        with amp_ctx():
                            with torch.no_grad():
                                p_fake = torch.sigmoid(self.student(x_u)[0].float())
                            p_real = y_l.float()
                            d_real = self.discriminator(x_l.float(), p_real)
                            d_fake = self.discriminator(x_u.float(), p_fake.detach())
                            d_loss = self.lambda_d * gan_discriminator_loss(d_real, d_fake)

                        self.scaler_d.scale(d_loss).backward()
                        self.scaler_d.unscale_(self.optim_d)
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                        self.scaler_d.step(self.optim_d)
                        self.scaler_d.update()
                        self.optim_d.zero_grad(set_to_none=True)

            running += _safe_scalar(g_loss * self.grad_accum_steps)
            conf_running += _safe_scalar(pseudo_w.mean())
            rel_running += _safe_scalar(reliability.mean())
            ood_running += _safe_scalar(rel_parts["ood_map"].mean())
            cons_running += _safe_scalar(rel_parts["consistency_map"].mean())
            n_steps += 1

        d = max(1, n_steps)
        return running / d, conf_running / d, rel_running / d, ood_running / d, cons_running / d

    @torch.no_grad()
    def evaluate(self, loader, use_teacher_ema: bool = True, use_ensemble: bool = False):
        self.student.eval()
        self.teacher.eval()

        agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "minority_f1": 0.0, "hd95": 0.0}
        n = 0

        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            with (torch.amp.autocast("cuda") if self.use_amp else nullcontext()):
                if use_ensemble:
                    z1, _ = self.student(x)
                    z2, _ = self.teacher(x)
                    logits = 0.5 * z1 + 0.5 * z2
                else:
                    if use_teacher_ema:
                        logits, _ = self.teacher(x)
                    else:
                        logits, _ = self.student(x)

            m = compute_binary_metrics(logits.float(), y.float(), threshold=float(self.cfg["inference"].get("threshold", 0.40)))
            agg["dice"] += max(0.0, min(1.0, float(m.dice)))
            agg["iou"] += max(0.0, min(1.0, float(m.iou)))
            agg["precision"] += max(0.0, min(1.0, float(m.precision)))
            agg["recall"] += max(0.0, min(1.0, float(m.recall)))
            agg["f1"] += max(0.0, min(1.0, float(m.f1)))
            agg["minority_f1"] += max(0.0, min(1.0, float(m.minority_f1)))
            hd = float(m.hd95)
            agg["hd95"] += 1e3 if not torch.isfinite(torch.tensor(hd)) else hd
            n += 1

        from utils.metrics import SegMetrics
        d = max(1, n)
        return SegMetrics(
            agg["dice"] / d, agg["iou"] / d, agg["precision"] / d,
            agg["recall"] / d, agg["f1"] / d, agg["minority_f1"] / d, agg["hd95"] / d
        )
    