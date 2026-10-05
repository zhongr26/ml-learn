import lightning as L
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as T

from config import (MNIST_DATA_DIR, BATCH_SIZE, NUM_WORKERS)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir=MNIST_DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        self.eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, download=True, train=True)
        datasets.MNIST(self.data_dir, download=True, train=False)

    def setup(self, stage=None):
        if stage in ("fit", None):
            full = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)

            self.train_set, self.val_set = data.random_split(full, [55000, 5000])
            self.val_set.dataset.transform = self.eval_transform

        if stage in ("test", "predict", None):
            self.test_set = datasets.MNIST(
                self.data_dir, train=False, transform=self.eval_transform
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
