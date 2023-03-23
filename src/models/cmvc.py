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

        loss = 0.01 * uttr_rc
        loss += face_rc
        loss += 0.001 * voice_rc
        loss += 0.01 * uttr_kl
        loss += face_kl

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/uttr", uttr_rc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/face", face_rc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.net.loss_function(x, y)
        loss = 0.01 * uttr_rc
        loss += face_rc
        loss += 0.001 * voice_rc
        loss += 0.01 * uttr_kl
        loss += face_kl

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx: int):
        pass

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
