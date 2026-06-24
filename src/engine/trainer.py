import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from src.losses.seg_losses import supervised_loss, unsupervised_loss


class MeanTeacherTrainer:

    def __init__(self, model, cfg):

        self.student = model.cuda()
        self.teacher = model.cuda()

        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()

        self.opt = Adam(self.student.parameters(), lr=1e-4)
        self.scaler = GradScaler("cuda")

        self.ema = 0.99

    @torch.no_grad()
    def update_teacher(self):

        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(self.ema).add_(s.data, alpha=1-self.ema)

    def train_step(self, batch_l, batch_u):

        x_l = batch_l["image"].cuda().float()
        y_l = batch_l["label"].cuda().float()
        x_u = batch_u["image"].cuda().float()

        with autocast("cuda"):

            s_l = self.student(x_l)
            s_u = self.student(x_u)

            with torch.no_grad():
                t_u = self.teacher(x_u)

            pseudo = torch.sigmoid(t_u)

            loss_sup = supervised_loss(s_l, y_l)
            loss_unsup = unsupervised_loss(s_u, pseudo)

            loss = loss_sup + 0.3 * loss_unsup

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        self.update_teacher()

        return loss.item()