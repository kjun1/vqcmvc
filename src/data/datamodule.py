from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from src.data.dataset import CrossDataset


class CrossJVSDataModule(LightningDataModule):

    def __init__(
        self,
        audio_data_dir: str = "36_40_train_mc/",
        image_data_dir: str = "images/",
        train_val_test_split: list = [0.8, 0.2],
        batch_size: int = 128,
        num_workers: int = 24,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.audio_data_dir = audio_data_dir
        self.image_data_dir = image_data_dir
        self.batch_size = batch_size
        self.num_workers = 24

        self.data_train = None
        self.data_valid = None
        self.data_test = None

    def setup(self, stage):
        if not self.data_train and not self.data_valid:
            dataset = CrossDataset(audio_data_dir=self.audio_data_dir, image_data_dir=self.image_data_dir)

            self.data_train, self.data_valid, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
