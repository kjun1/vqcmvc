import math
from typing import Union

import torch
from torch import nn

from .activation import GLU


class UttrEncoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.model['lr'] = GLU()
        self.num_layers = params['num_layers']

        for i in range(1, self.num_layers-1):
            self.model[f'conv{i}a'] = nn.Conv2d(
                params[f'conv{i-1}_channels'],
                params[f'conv{i}_channels'],
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                conv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']
                )

            self.model[f'conv{i}b'] = nn.Conv2d(
                params[f'conv{i-1}_channels'],
                params[f'conv{i}_channels'],
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                conv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='replicate'
            )

            self.model[f'bn{i}b'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']
                )

        self.model[f'conv{self.num_layers}'] = nn.Conv2d(
            params[f'conv{self.num_layers-1}_channels'],
            params[f'conv{self.num_layers}_channels'],
            params[f'conv{self.num_layers}_kernel'],
            params[f'conv{self.num_layers}_stride'],
            conv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
            bias=False, padding_mode='replicate'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(1, self.num_layers):
            x1 = self.model[f'conv{i}a'](x)
            x1 = self.model[f'bn{i}a'](x1)
            x2 = self.model[f'conv{i}b'](x)
            x2 = self.model[f'bn{i}b'](x1)
            x = self.model['lr'](x1, x2)
        x = self.model[f'conv{self.num_layers}'](x)
        return x

    def test_input(self) -> None:
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x).shape)


def conv_padding(kernel: int | tuple, stride: int | tuple) -> int | list:
    assert type(kernel) == type(stride), "type error"
    if type(kernel) == int:
        return math.floor((kernel - stride)/2)
    else:
        padding = []
        for k, s in zip(kernel, stride):
            padding.append(conv_padding(k, s))
        return padding


def deconv_padding(kernel: int | tuple, stride: int | tuple) -> tuple:
    assert type(kernel) == type(stride), "type error"
    if type(kernel) == int:
        return math.ceil((kernel - stride)/2), (kernel - stride) % 2
    else:
        padding = []
        output_padding = []
        for k, s in zip(kernel, stride):
            p, q = deconv_padding(k, s)
            padding.append(p)
            output_padding.append(q)
        return padding, output_padding


class UttrDecoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self.model['lr'] = GLU()
            self.num_layers = params['num_layers']
            self.num_layers = params['linear_layers']
            self.num_layers = params['conv_layers']
            self._init_model(params)

        for i in range(1, self.num_layers-1):
            self.model[f'deconv{i}a'] = nn.ConvTranspose2d(
                params[f'conv{i-1}_channels']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
                self.hparams[f'UTTR_DEC_CONV{i}_CHANNELS'],
                self.hparams[f'UTTR_DEC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_DEC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_DEC_CONV{i}_PADDING'],
                bias=False, padding_mode='zeros'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'])

            self.model[f'deconv{i}b'] = nn.ConvTranspose2d(
                self.hparams[f'UTTR_DEC_CONV{i}_CHANNELS']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
                self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_DEC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_DEC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_DEC_CONV{i}_PADDING'],
                self.hparams[f'UTTR_DEC_CONV{i}_OUT_PADDING'],
                bias=False, padding_mode='zeros'
            )
            self.model[f'bn{i}b'] = nn.BatchNorm2d(self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'])

        self.model[f'deconv{NUM_LAYERS}'] =  nn.ConvTranspose2d(
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_CHANNELS']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='zeros'
        )

    def forward(self, x, y):
        NUM_LAYERS= self.hparams['UTTR_DEC_NUM_LAYERS']
        c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
        z = torch.cat((x, c), dim=1)
        for i in range(1, NUM_LAYERS):
            z1 = self.model[f'deconv{i}a'](z)
            z1 = self.model[f'bn{i}a'](z1)
            z2 = self.model[f'deconv{i}b'](z)
            z2 = self.model[f'bn{i}b'](z1)
            x = self.model['lr'](z1, z2)
            c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
            z = torch.cat((x, c), dim=1)
        x = self.model[f'deconv{NUM_LAYERS}'](z)
        return x
