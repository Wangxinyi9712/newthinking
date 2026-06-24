import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from src.models.bayes_teacher import BayesianTeacher
from src.losses.seg_losses import *
from src.losses.contrastive import prototype_contrast_loss
from src.utils.frequency import frequency_filter


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.student = model.cuda()
        self.teacher = BayesianTeacher(model).cuda()

        self.opt = Adam(self.student.parameters(), lr=1e-4)

        self.scaler = GradScaler("cuda")
        self.ema = 0.99

        self.memory = {}

    def update_teacher(self):

        for t, s in zip(self.teacher.model.parameters(), self.student.parameters()):
            t.data = self.ema * t.data + (1 - self.ema) * s.data

    def train_step(self, batch_l, batch_u):

        x_l = batch_l["image"].cuda()
        y_l = batch_l["label"].cuda()
        x_u = batch_u["image"].cuda()

        with autocast("cuda"):

            s_l, feat_l = self.student(x_l, return_features=True)
            s_u, feat_u = self.student(x_u, return_features=True)

            t_mean, t_var = self.teacher(x_u)

            pseudo = frequency_filter(t_mean)

            loss = (
                supervised_loss(s_l, y_l) +
                0.5 * unsupervised_loss(s_u, pseudo) +
                0.1 * spectral_consistency_loss(s_u, t_mean) +
                0.05 * prototype_contrast_loss(feat_l, y_l, self.memory)
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        self.update_teacher()

        return loss.item()