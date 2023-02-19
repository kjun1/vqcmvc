import hydra
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from src.models.components.block import conv_padding, deconv_padding


class TestNet:
    def test_init(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            assert cfg.net  # attribute style access
            assert cfg["net"]  # dictionary style access

    def test_conv(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)

            assert cfg.net.uttrenc.params.conv1_channels
            assert cfg.net.uttrenc.params.conv1_kernel
            assert cfg.net.uttrenc.params.conv1_stride

    def test_instantiate_1(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc copy.yaml", return_hydra_config=True)
            assert hydra.utils.instantiate(cfg.net.uttrenc)

    def test_instantiate_2(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc copy.yaml", return_hydra_config=True)
            assert hydra.utils.instantiate(cfg.net)


class TestPadding:
    """
    pytestの設定をいれたい
    """
    def test_conv_padding_int(self) -> None:
        conv_padding(1, 1)

    def test_conv_padding_tuple(self) -> None:
        conv_padding((1, 1), (1, 1))

    def test_deconv_padding_int(self) -> None:
        deconv_padding(1, 1)

    def test_deconv_padding_tuple(self) -> None:
        deconv_padding((1, 1), (1, 1))
