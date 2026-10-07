import torch
import torch.nn as nn

from config import MNIST_CFG
from .model import LitAutoModel


class _AutoEncoder(nn.Module):
    """纯 nn.Module 包装，供 TorchScript 导出（LightningModule 含 Trainer 相关逻辑，无法直接 script）。"""

    def __init__(self, model: LitAutoModel):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def save_and_load_demo(ckpt_path):
    model = LitAutoModel.load_from_checkpoint(ckpt_path)
    model.eval()

    state_path = MNIST_CFG.checkpoint_dir / "mnist_ae_state.pt"
    scripted_path = MNIST_CFG.checkpoint_dir / "mnist_ae_scripted.pt"

    torch.save(model.state_dict(), state_path)
    torch.jit.save(torch.jit.script(_AutoEncoder(model).eval()), scripted_path)

    new_model = LitAutoModel(latent_dim=MNIST_CFG.hparams.latent_dim)
    new_model.load_state_dict(torch.load(state_path, weights_only=True))
    new_model.eval()
    print("模型加载成功。")

    torch.jit.load(scripted_path)
    print("TorchScript 加载成功。")
    return model
