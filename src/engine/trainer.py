import torch
import csv
from pathlib import Path
from copy import deepcopy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.losses.seg_losses import (
    supervised_loss,
    spectral_consistency_loss,
    cps_loss,
    topology_loss,
    unsupervised_loss,
    feature_contrastive_loss,
)
from src.utils.metrics import compute_binary_metrics


@torch.no_grad()
def update_ema(student, teacher, base_m=0.99):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(base_m).add_(s.data * (1 - base_m))


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = model.to(self.device)
        self.teacher = deepcopy(model).to(self.device)

        self.optim = Adam(self.student.parameters(), lr=cfg.train["lr"])
        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=cfg.train["epochs"],
            eta_min=cfg.train["min_lr"]
        )

        self.epochs = cfg.train["epochs"]
        self.best_dice = -1.0

        self.lambda_ssl = cfg.loss.get("lambda_ssl", 1.0)
        self.lambda_cps = cfg.loss.get("lambda_cps", 0.3)
        self.lambda_topo = cfg.loss.get("lambda_topo", 0.1)
        self.lambda_feat = cfg.loss.get("lambda_feat", 0.05)
        self.lambda_spec = cfg.loss.get("lambda_spec", 0.2)

    def fit(self, loaders, out_dir):

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):

            self.student.train()

            pbar = tqdm(loaders["labeled"], desc=f"epoch {epoch}")

            for batch in pbar:

                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)

                xu = next(iter(loaders["unlabeled"]))["image"].to(self.device)

                sl, fl = self.student(x, return_features=True)
                su, fu = self.student(xu, return_features=True)

                with torch.no_grad():
                    tl, ft = self.teacher(xu, return_features=True)
                    pseudo = torch.sigmoid(tl)

                loss = (
                    supervised_loss(sl, y)
                    + self.lambda_ssl * unsupervised_loss(su, pseudo)
                    + self.lambda_cps * cps_loss(sl, sl)
                    + self.lambda_topo * topology_loss(sl, y)
                    + self.lambda_feat * feature_contrastive_loss(fu, ft)
                    + self.lambda_spec * spectral_consistency_loss(su, tl)
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                update_ema(self.student, self.teacher)

                pbar.set_postfix(loss=float(loss))

            self.scheduler.step()

            # =========================
            # evaluation (light)
            # =========================
            self.student.eval()

            dice_sum = 0
            n = 0

            for batch in loaders["val"]:

                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)

                logits, _ = self.student(x, return_features=True)

                m = compute_binary_metrics(logits, y)

                dice_sum += m.dice
                n += 1

            dice = dice_sum / max(1, n)

            # =========================
            # checkpoint policy (STRICT)
            # =========================
            ckpt = {
                "student": self.student.state_dict(),
                "teacher": self.teacher.state_dict(),
                "epoch": epoch,
                "dice": dice
            }

            torch.save(ckpt, out_dir / "last.pt")

            if dice > self.best_dice:
                self.best_dice = dice
                torch.save(ckpt, out_dir / "best.pt")

            print(f"[Epoch {epoch}] dice={dice:.4f} best={self.best_dice:.4f}")