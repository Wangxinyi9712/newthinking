import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.models.bayes_teacher import BayesianTeacher
from src.models.diffusion_refiner import DiffusionRefiner
from src.losses.seg_losses import (
    supervised_loss,
    unsupervised_loss,
    spectral_consistency_loss,
    prototype_contrast_loss,
)


class MeanTeacherTrainer:

    def __init__(self, model, cfg):
        self.student = model.cuda()
        self.teacher = BayesianTeacher(model).cuda()
        self.refiner = DiffusionRefiner().cuda()

        self.opt = Adam(self.student.parameters(), lr=1e-4)

        self.ema_base = cfg.train.get("ema_momentum", 0.99)

    def _uncertainty_ema(self, var_map):
        """
        α(x) = normalized uncertainty
        """
        u = var_map.mean()
        alpha = torch.sigmoid(u)
        return alpha

    @torch.no_grad()
    def update_teacher(self, var_map):
        alpha = self._uncertainty_ema(var_map)

        for t, s in zip(self.teacher.model.parameters(), self.student.parameters()):
            t.data = alpha * t.data + (1 - alpha) * s.data

    def train_step(self, batch_l, batch_u):

        x_l, y_l = batch_l["image"].cuda(), batch_l["label"].cuda()
        x_u = batch_u["image"].cuda()

        # student
        s_l = self.student(x_l)
        s_u = self.student(x_u)

        # teacher (Bayesian)
        t_mean, t_var = self.teacher(x_u)

        # diffusion refine
        pseudo = self.refiner(t_mean)

        # spectral filter
        pseudo = pseudo + 0.1 * torch.randn_like(pseudo)

        # losses
        loss_sup = supervised_loss(s_l, y_l)
        loss_unsup = unsupervised_loss(s_u, pseudo)
        loss_spec = spectral_consistency_loss(s_u, t_mean)

        loss_proto = prototype_contrast_loss(
            self.student(x_l, True)[1], y_l
        )

        loss = (
            loss_sup
            + loss_unsup
            + 0.1 * loss_spec
            + 0.2 * loss_proto
        )

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_teacher(t_var)

        return loss.item()