import lightning as L
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as T

from config import (MNIST_DATA_DIR, MNIST_HPARAMS)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir=MNIST_DATA_DIR,
                 batch_size=MNIST_HPARAMS["batch_size"],
                 num_workers=MNIST_HPARAMS["num_workers"]):
        super().__init__()

        # initialized in self.setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1)),
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
            # 输入归一化到 [-1,1]，解码器用 Tanh
            T.Normalize((0.5,), (0.5,))
        ])
        self.eval_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,))
            T.Normalize((0.5,), (0.5,))
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

    def _loader(self, dataset, shuffle):
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return data.DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()
