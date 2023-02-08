from typing import Union

import pytorch_lightning as pl
import torch
from torch import nn

from .activation import GLU


class UttrEncoder(pl.LightningModule):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.save_hyperparameters(params)
        self.model['lr'] = GLU()

        NUM_LAYERS = self.hparams['UTTR_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            self.model[f'conv{i}a'] = nn.Conv2d(
                self.hparams[f'UTTR_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_ENC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS']
                )

            self.model[f'conv{i}b'] = nn.Conv2d(
                self.hparams[f'UTTR_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_ENC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )

            self.model[f'bn{i}b'] = nn.BatchNorm2d(
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS']
                )

        self.model[f'conv{NUM_LAYERS}'] = nn.Conv2d(
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_CHANNELS'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='replicate'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        NUM_LAYERS = self.hparams['UTTR_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            x1 = self.model[f'conv{i}a'](x)
            x1 = self.model[f'bn{i}a'](x1)
            x2 = self.model[f'conv{i}b'](x)
            x2 = self.model[f'bn{i}b'](x1)
            x = self.model['lr'](x1, x2)
        x = self.model[f'conv{NUM_LAYERS}'](x)
        return x

    def test_input(self) -> None:
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x).shape)
