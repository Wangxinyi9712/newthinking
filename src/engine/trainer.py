import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from src.losses.seg_losses import *
from src.losses.contrastive import prototype_contrast_loss
from src.utils.frequency import frequency_filter


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.student = model.cuda()
        self.teacher = model.cuda()
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()

        self.opt = Adam(self.student.parameters(), lr=1e-4)
        self.scaler = GradScaler("cuda")

        self.ema = 0.99

        self.memory = {}

    @torch.no_grad()
    def update_teacher(self):
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data = self.ema * t.data + (1 - self.ema) * s.data

    def train_step(self, batch_l, batch_u):

        x_l = batch_l["image"].cuda()
        y_l = batch_l["label"].cuda()
        x_u = batch_u["image"].cuda()

        with autocast("cuda"):

            s_l, feat_l = self.student(x_l, return_features=True)
            s_u, feat_u = self.student(x_u, return_features=True)

            with torch.no_grad():
                t_u = self.teacher(x_u)

            pseudo = frequency_filter(torch.sigmoid(t_u))

            loss_sup = supervised_loss(s_l, y_l)
            loss_unsup = unsupervised_loss(s_u, pseudo)
            loss_spec = spectral_consistency_loss(s_u, t_u)

            loss_proto = prototype_contrast_loss(feat_l, y_l, self.memory)

            loss = loss_sup + 0.5 * loss_unsup + 0.1 * loss_spec + 0.1 * loss_proto

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        self.update_teacher()

        return loss.item()