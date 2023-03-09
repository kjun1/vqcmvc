import pytorch_lightning as pl
import torch
from torch import nn


class CrossModalLitModule(pl.LightningModule):
    def __init__(self, net, optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

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

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        return {"optimizer": optimizer}
