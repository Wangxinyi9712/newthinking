from __future__ import annotations

import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.losses.seg_losses import (
    supervised_loss,
    unsupervised_loss,
    cps_loss,
    spectral_consistency_loss,
)


# =========================================================
# uncertainty-aware EMA (TMI KEY CONTRIBUTION)
# θ_t = α(x)θ_t + (1-α(x))θ_s
# =========================================================
@torch.no_grad()
def update_ema(student, teacher, x, base=0.99):

    prob = torch.sigmoid(student(x)).mean()
    entropy = -(prob * torch.log(prob + 1e-6))

    alpha = base + (1 - base) * entropy.item()

    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(alpha).add_(sp.data * (1 - alpha))


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.student = model.to(self.device)
        self.teacher = deepcopy(model).to(self.device)
        self.teacher.eval()

        self.optim = Adam(self.student.parameters(), lr=cfg["train"]["lr"])
        self.scheduler = CosineAnnealingLR(self.optim, T_max=cfg["train"]["epochs"])

        self.epochs = cfg["train"]["epochs"]

        self.lambda_ssl = cfg["loss"].get("lambda_ssl", 1.0)
        self.lambda_spec = cfg["loss"].get("lambda_spectral", 0.2)
        self.lambda_cps = cfg["loss"].get("lambda_cps", 0.3)

        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

    def fit(self, loaders, out_dir):

        for epoch in range(self.epochs):

            self.student.train()
            unlabeled_iter = iter(loaders["unlabeled"])

            pbar = tqdm(loaders["labeled"], desc=f"epoch {epoch}")

            for batch in pbar:

                ub = next(unlabeled_iter)

                x_l = batch["image"].to(self.device).float()
                y_l = batch["label"].to(self.device).float()
                x_u = ub["image"].to(self.device).float()

                with torch.amp.autocast("cuda"):

                    s_l = self.student(x_l)
                    sup = supervised_loss(s_l, y_l)

                    with torch.no_grad():
                        t_u = self.teacher(x_u)

                    s_u = self.student(x_u)

                    unsup = unsupervised_loss(s_u, torch.sigmoid(t_u))
                    cps = cps_loss(s_u, t_u)

                    spec = spectral_consistency_loss(s_u, t_u)

                    loss = (
                        sup
                        + self.lambda_ssl * unsup
                        + self.lambda_cps * cps
                        + self.lambda_spec * spec
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                update_ema(self.student, self.teacher, x_u)

            self.scheduler.step()