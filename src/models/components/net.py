import torch
from torch import nn


class CrossModal(nn.Module):
    def __init__(self, uttrenc):
        super().__init__()
        self.uttrenc = uttrenc

    def forward(self, x, y):
        z, _ = torch.chunk(self.ue(x), 2, dim=1)
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        x, _ = torch.chunk(self.ud(z, c), 2, dim=1)
        return x
