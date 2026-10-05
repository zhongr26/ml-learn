import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid


class LitAutoModel(L.LightningModule):
    def __init__(self, latent_dim=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def _shared_step(self, batch, stage):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)  # (1)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x, x_hat)  # (2)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        return loss, x, x_hat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x, x_hat = self._shared_step(batch, stage='val')
        if batch_idx == 0:
            x_hat = x_hat.view(-1, 1, 28, 28)
            self._log_reconstructions(x, x_hat, "val_recon")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage='test')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28), x, z

    def _log_reconstructions(self, x, x_hat, name):
        # 防御性 reshape：兼容 (B, 784) 与 (B, 1, 28, 28)
        if x_hat.dim() == 2:
            x_hat = x_hat.view(-1, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        n = min(8, x.size(0))
        grid = make_grid(
            torch.cat([x[:n], x_hat[:n]], dim=0),
            nrow=n, normalize=True,
        )
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
