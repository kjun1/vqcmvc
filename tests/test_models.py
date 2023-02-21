import pytest

from src.models.components.net import UttrEncoder


def test_load_checkpoint() -> None:
    import torch
    checkpoint = torch.load("last.ckpt", map_location=lambda storage, loc: storage)
