import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * torch.sigmoid(x2)
