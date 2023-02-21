import math
from typing import Union

import torch
from torch import nn


def conv_padding(kernel: int | tuple, stride: int | tuple) -> dict:
    assert type(kernel) == type(stride), "type error"
    if type(kernel) == int:
        return math.floor((kernel - stride)/2)
    else:
        padding = []
        for k, s in zip(kernel, stride):
            padding.append(conv_padding(k, s))
        return {"padding": padding}


def deconv_padding(kernel: int | tuple, stride: int | tuple) -> dict:
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
        return {"padding": padding, "output_padding": output_padding}


class UttrEncoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.model['activation'] = nn.GLU(dim=1)
        self.num_layers = params['num_layers']

        for i in range(1, self.num_layers):
            self.model[f'conv{i}'] = nn.Conv2d(
                params[f'conv{i-1}_channels'],
                params[f'conv{i}_channels']*2,
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                **conv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']*2
                )

        self.model[f'conv{self.num_layers}'] = nn.Conv2d(
            params[f'conv{self.num_layers-1}_channels'],
            params[f'conv{self.num_layers}_channels'],
            params[f'conv{self.num_layers}_kernel'],
            params[f'conv{self.num_layers}_stride'],
            **conv_padding(
                    params[f'conv{self.num_layers}_kernel'],
                    params[f'conv{self.num_layers}_stride']
                    ),
            bias=False, padding_mode='replicate'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(1, self.num_layers):
            x = self.model[f'conv{i}'](x)
            x = self.model[f'bn{i}'](x)
            x = self.model['activation'](x)
        x = self.model[f'conv{self.num_layers}'](x)
        return x


class UttrDecoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.model['activation'] = nn.GLU(dim=1)
        self.num_layers = params['num_layers']

        for i in range(1, self.num_layers):
            self.model[f'deconv{i}'] = nn.ConvTranspose2d(
                params[f'conv{i-1}_channels']+params['latent_channels'],
                params[f'conv{i}_channels']*2,
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                **deconv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='zeros'
                )
            self.model[f'bn{i}'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']*2
                )

        self.model[f'deconv{self.num_layers}'] = nn.ConvTranspose2d(
            params[f'conv{self.num_layers-1}_channels']+params['latent_channels'],
            params[f'conv{self.num_layers}_channels'],
            params[f'conv{self.num_layers}_kernel'],
            params[f'conv{self.num_layers}_stride'],
            **deconv_padding(
                params[f'conv{self.num_layers}_kernel'],
                params[f'conv{self.num_layers}_stride']
                ),
            bias=False, padding_mode='zeros'
            )

    def forward(self, x, y):
        for i in range(1, self.num_layers):
            c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
            z = torch.cat((x, c), dim=1)
            z = self.model[f'deconv{i}'](z)
            z = self.model[f'bn{i}'](z)
            x = self.model['activation'](z)

        c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
        z = torch.cat((x, c), dim=1)
        x = self.model[f'deconv{self.num_layers}'](z)
        return x


class FaceEncoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.model['activation'] = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.num_layers = params['num_layers']
        self.conv_layers = params['conv_layers']
        self.linear_layers = params['linear_layers']

        for i in range(1, self.conv_layers+1):
            self.model[f'conv{i}'] = nn.Conv2d(
                params[f'conv{i-1}_channels'],
                params[f'conv{i}_channels'],
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                **conv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']
                )

        for i in range(1, self.linear_layers+1):
            self.model[f'linear{i}'] = nn.Linear(
                params[f'linear{i-1}_size'],
                params[f'linear{i}_size'],
                bias=False,
            )

    def forward(self, x):
        for i in range(1, self.conv_layers+1):
            x = self.model[f'conv{i}'](x)
            if i != 1:
                x = self.model[f'bn{i}'](x)
            x = self.model['activation'](x)
        x = x.view(x.shape[0], -1)

        for i in range(1, self.linear_layers+1):
            x = self.model[f'linear{i}'](x)
            x = self.model['activation'](x)

        x = x.unsqueeze(-1).unsqueeze(-1)
        return x


class FaceDecoder(nn.Module):
    def __init__(self, params: Union[dict, None] = None) -> None:
        super().__init__()
        self.model = nn.ModuleDict()
        if params is not None:
            self._init_model(params)

    def _init_model(self, params: dict) -> None:
        self.model['activation'] = torch.nn.Softplus()
        self.num_layers = params['num_layers']
        self.conv_layers = params['conv_layers']
        self.linear_layers = params['linear_layers']

        for i in range(1, self.linear_layers+1):
            self.model[f'linear{i}'] = nn.Linear(
                params[f'linear{i-1}_size'],
                params[f'linear{i}_size'],
                bias=False,
            )


        for i in range(1, self.conv_layers):
            self.model[f'deconv{i}'] = nn.ConvTranspose2d(
                params[f'conv{i-1}_channels'],
                params[f'conv{i}_channels'],
                params[f'conv{i}_kernel'],
                params[f'conv{i}_stride'],
                **deconv_padding(
                    params[f'conv{i}_kernel'],
                    params[f'conv{i}_stride']
                    ),
                bias=False, padding_mode='zeros'
                )
            self.model[f'bn{i}'] = nn.BatchNorm2d(
                params[f'conv{i}_channels']
                )

        self.model[f'deconv{self.conv_layers}'] = nn.Conv2d(
            params[f'conv{self.conv_layers-1}_channels'],
            params[f'conv{self.conv_layers}_channels'],
            params[f'conv{self.conv_layers}_kernel'],
            params[f'conv{self.conv_layers}_stride'],
            **conv_padding(
                    params[f'conv{self.conv_layers}_kernel'],
                    params[f'conv{self.conv_layers}_stride']
                    ),
            bias=False, padding_mode='replicate'
        )

    def forward(self, x):
        for i in range(1, self.linear_layers+1):
            x = self.model[f'linear{i}'](x)
            x = self.model['activation'](x)
        h = int(math.sqrt(self.model[f'linear{self.linear_layers}'].out_features / self.model['deconv1'].in_channels))
        print(x.shape)
        x = x.view(x.shape[0], self.model['deconv1'].in_channels, h, h)

        for i in range(1, self.conv_layers):
            x = self.model[f'deconv{i}'](x)
            if i != self.conv_layers:
                x = self.model[f'bn{i}'](x)
            x = self.model['activation'](x)
        x = self.model[f'deconv{self.conv_layers}'](x)

        return x

class VoiceEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)

        self.model['lr'] = GLU()

        NUM_LAYERS = self.hparams['VOICE_ENC_NUM_LAYERS']

        for i in range(1, NUM_LAYERS):
            self.model[f'conv{i}a'] = nn.Conv2d(
                self.hparams[f'VOICE_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i}_KERNEL'],
                self.hparams[f'VOICE_ENC_CONV{i}_STRIDE'],
                self.hparams[f'VOICE_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'])

            self.model[f'conv{i}b'] = nn.Conv2d(
                self.hparams[f'VOICE_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i}_KERNEL'],
                self.hparams[f'VOICE_ENC_CONV{i}_STRIDE'],
                self.hparams[f'VOICE_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}b'] = nn.BatchNorm2d(self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'])

        self.model[f'conv{NUM_LAYERS}'] =  nn.Conv2d(
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_CHANNELS'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='replicate'
        )



    def forward(self, x):
        NUM_LAYERS = self.hparams['VOICE_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            x1 = self.model[f'conv{i}a'](x)
            x1 = self.model[f'bn{i}a'](x1)
            x2 = self.model[f'conv{i}b'](x)
            x2 = self.model[f'bn{i}b'](x1)
            x = self.model['lr'](x1, x2)
        x = self.model[f'conv{NUM_LAYERS}'](x)
        return x


class CrossModal(nn.Module):
    def __init__(self, uttrenc, uttrdec, faceenc, facedec, vonceenc):
        super().__init__()
        self.uttrenc = uttrenc
        self.uttrdec = uttrdec
        self.faceenc = faceenc
        self.facedec = facedec

    def forward(self, x, y):
        z, _ = torch.chunk(self.ue(x), 2, dim=1)
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        x, _ = torch.chunk(self.ud(z, c), 2, dim=1)
        return x
