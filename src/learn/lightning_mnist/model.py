import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid


# ============================================================
# SSIM 损失
# ============================================================
def ssim_loss(x, y, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """结构相似性损失 (1 - SSIM)。x, y: (B, C, H, W)，值域 [0,1]。"""
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)
    mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, 1, pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, window_size, 1, pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return 1 - ssim_map.mean()


# ============================================================
# 卷积编码器
# ============================================================
class ConvEncoder(nn.Module):
    """(B,1,28,28) → (B, latent_dim)"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.fc(h)


# ============================================================
# 卷积解码器
# ============================================================
class ConvDecoder(nn.Module):
    """(B, latent_dim) → (B,1,28,28)"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7→14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14→28
            # nn.Sigmoid(),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.net(h)


# ============================================================
# 组合为 Lightning 模块
# ============================================================
class LitAutoModel(L.LightningModule):
    def __init__(self, latent_dim=3, lr=1e-3, ssim_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.lr = lr
        self.ssim_weight = ssim_weight

    def forward(self, x):
        return self.encoder(x)

    def _shared_step(self, batch, stage):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)  # (1)

        l1 = F.l1_loss(x, x_hat)
        ssim = ssim_loss(x, x_hat)
        loss = l1 + self.ssim_weight * ssim  # (2)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_l1", l1, on_epoch=True)
        self.log(f"{stage}_ssim", ssim, on_epoch=True)

        return loss, x, x_hat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x, x_hat = self._shared_step(batch, stage='val')
        if batch_idx == 0:
            self._log_reconstructions(x, x_hat, "val_recon")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage='test')
        return loss

    def _log_reconstructions(self, x, x_hat, name):
        n = min(8, x.size(0))
        grid = make_grid(
            torch.cat([x[:n], x_hat[:n]], dim=0),
            nrow=n, normalize=True,
        )
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
