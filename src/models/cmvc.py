import pytorch_lightning as pl
import torch
from torch import nn


class CrossModalLitModule(pl.LightningModule):
    def __init__(self, net, optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

    def forward(self, x, y):
        return self.net(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.net.loss_function(x, y)
        uttr_loss = uttr_rc
        uttr_loss += uttr_kl

        face_loss = face_rc
        face_loss += voice_rc
        face_loss += face_kl

        train_loss = uttr_loss + face_loss

        self.log("train_loss", train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.net.loss_function(x, y)
        loss = uttr_rc
        loss += face_rc
        loss += voice_rc
        loss += uttr_kl
        loss += face_kl

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
