import pytorch_lightning as pl
import torch
from torch import nn


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = kwargs["net"]

    def forward(self, x, y):
        z, _ = torch.chunk(self.ue(x), 2, dim=1)
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        x, _ = torch.chunk(self.ud(z, c), 2, dim=1)
        return x

    def rc_image(self, y):
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        y, _ = torch.chunk(self.fd(c.squeeze(-1).squeeze(-1)), 2, dim=1)
        return y.to(torch.uint8).squeeze(0)

    def loss_function(self, x, y):
        mu, log_var = torch.chunk(self.ue(x), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_kl = self._KL_divergence(mu, log_var)
        z = self._sample_z(mu, log_var)

        mu, log_var = torch.chunk(self.fe(y), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        face_kl = self._KL_divergence(mu, log_var)
        c = self._sample_z(mu, log_var)

        mu, log_var = torch.chunk(self.ud(z, c), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_rc = self._reconstruction(x, mu, log_var)
        x_hat = self._sample_z(mu, log_var)

        mu, log_var = torch.chunk(self.fd(c.squeeze(-1).squeeze(-1)), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        face_rc = self._reconstruction(y, mu, log_var)

        mu, log_var = torch.chunk(self.ve(x_hat), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        voice_rc = []

        for i, j in zip(
            torch.tensor_split(mu, mu.shape[-1], dim=-1),
            torch.tensor_split(log_var, log_var.shape[-1], dim=-1)
        ):
            voice_rc.append(self._reconstruction(c, i, j))

        voice_rc = torch.sum(torch.stack(voice_rc)).to(self.device)/len(voice_rc)

        return uttr_rc, face_rc, voice_rc, uttr_kl, face_kl

    def training_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.loss_function(x, y)
        uttr_loss = self.hparams["LAMBDA1"]*uttr_rc
        uttr_loss += self.hparams["LAMBDA4"]*uttr_kl

        face_loss = self.hparams["LAMBDA2"]*face_rc
        face_loss += self.hparams["LAMBDA3"]*voice_rc
        face_loss += self.hparams["LAMBDA5"]*face_kl

        loss_schedule = self.current_epoch//200
        loss = uttr_loss/(2**loss_schedule) + face_loss/(4**loss_schedule)

        self.log("uttr_rc", uttr_rc)
        self.log("face_rc", face_rc)
        self.log("voice_rc", voice_rc)
        self.log("uttr_kl", uttr_kl)
        self.log("face_kl", face_kl)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.loss_function(x, y)
        loss = self.hparams["LAMBDA1"]*uttr_rc
        loss += self.hparams["LAMBDA2"]*face_rc
        loss += self.hparams["LAMBDA3"]*voice_rc
        loss += self.hparams["LAMBDA4"]*uttr_kl
        loss += self.hparams["LAMBDA5"]*face_kl

        self.log("val_loss", loss)

    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("encoder out mean_shape")
        print(self.forward(x, y).shape)

    def test_loss(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("output loss_function")
        print(self.loss_function(x, y))

    def _KL_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def _reconstruction(self, x, mu, log_var):
        return torch.sum(log_var + torch.square(x-mu)/log_var.exp())*0.5

    def _sample_z(self, mu, log_var):
        epsilon = torch.randn(mu.shape, device=self.device)
        return mu + log_var.exp() * epsilon

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["LR"])

        return [opt], []
