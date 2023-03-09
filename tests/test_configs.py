import hydra
import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig


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

    def test_uttrencoder(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            enc = hydra.utils.instantiate(cfg.net.uttrenc)
        x = torch.ones(64, 1, 36, 40)
        assert enc(x).shape == torch.Size([64, 16, 1, 10])

    def test_uttrdecoder(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            dec = hydra.utils.instantiate(cfg.net.uttrdec)
        x = torch.ones(64, 8, 1, 10)
        y = torch.ones(64, 8, 1, 1)
        assert dec(x, y).shape == torch.Size([64, 2, 36, 40])

    def test_faceencoder(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            enc = hydra.utils.instantiate(cfg.net.faceenc)
        x = torch.ones(64, 3, 32, 32)
        assert enc(x).shape == torch.Size([64, 16, 1, 1])

    def test_facedecoder(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            dec = hydra.utils.instantiate(cfg.net.facedec)
        x = torch.ones(64, 8)
        assert dec(x).shape == torch.Size([64, 6, 32, 32])

    def test_voiceencoder(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            enc = hydra.utils.instantiate(cfg.net.voiceenc)
        x = torch.ones(64, 1, 36, 40)
        assert enc(x).shape == torch.Size([64, 16, 1, 5])

    def test_instantiate(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="train.yaml", return_hydra_config=True)
            assert hydra.utils.instantiate(cfg.model)

    def test_crossmodal(self) -> None:
        with initialize(version_base=None, config_path="../configs/model"):
            cfg = compose(config_name="cmvc.yaml", return_hydra_config=True)
            net = hydra.utils.instantiate(cfg.net)
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        net.device = "cpu"
        print(net.loss_function(x, y))

    def test_data(self) -> None:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="train.yaml", return_hydra_config=True)
            assert hydra.utils.instantiate(cfg.data)
            datamodule = hydra.utils.instantiate(cfg.data)
            datamodule.setup(0)


"""
class TestPadding:
    def test_conv_padding_int(self) -> None:
        conv_padding(1, 1)

    def test_conv_padding_tuple(self) -> None:
        conv_padding((1, 1), (1, 1))

    def test_deconv_padding_int(self) -> None:
        deconv_padding(1, 1)

    def test_deconv_padding_tuple(self) -> None:
        deconv_padding((1, 1), (1, 1))
"""
