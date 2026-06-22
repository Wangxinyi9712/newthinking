from __future__ import annotations

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.models.bayes_teacher import BayesianTeacher
from src.models.diffusion_refiner import DiffusionRefiner
from src.losses.seg_losses import *
from src.losses.contrastive import prototype_contrast_loss
from src.utils.frequency import frequency_filter


class MeanTeacherTrainer:

    def __init__(self, model, cfg):
        self.student = model.cuda()
        self.teacher = BayesianTeacher(model).cuda()

        self.refiner = DiffusionRefiner().cuda()

        self.opt = Adam(self.student.parameters(), lr=1e-4)

        self.ema = 0.99

    def update_teacher(self):

        for t, s in zip(self.teacher.model.parameters(),
                        self.student.parameters()):

            t.data = self.ema * t.data + (1 - self.ema) * s.data

    def train_step(self, batch_l, batch_u):

        x_l, y_l = batch_l["image"].cuda(), batch_l["label"].cuda()
        x_u = batch_u["image"].cuda()

        # student
        s_l = self.student(x_l)
        s_u = self.student(x_u)

        # teacher Bayesian
        t_mean, t_var = self.teacher(x_u)

        # refine pseudo
        pseudo = self.refiner(t_mean)

        # frequency filter
        pseudo = frequency_filter(pseudo)

        loss_sup = supervised_loss(s_l, y_l)
        loss_unsup = unsupervised_loss(s_u, pseudo)

        loss_spec = spectral_consistency_loss(s_u, t_mean)

        loss = loss_sup + loss_unsup + 0.1 * loss_spec

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        self.update_teacher()

        return loss.item()