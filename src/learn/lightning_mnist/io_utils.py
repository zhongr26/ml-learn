import torch

from config import MNIST_CHECKPOINT_DIR, LATENT_DIM
from .model import LitAutoModel


def save_and_load_demo(ckpt_path):
    model = LitAutoModel.load_from_checkpoint(ckpt_path)
    model.eval()

    state_path = MNIST_CHECKPOINT_DIR / "mnist_ae_state.pt"
    scripted_path = MNIST_CHECKPOINT_DIR / "mnist_ae_scripted.pt"

    torch.save(model.state_dict(), state_path)
    torch.jit.save(model.to_torchscript(), scripted_path)

    new_model = LitAutoModel(latent_dim=LATENT_DIM)
    new_model.load_state_dict(torch.load(state_path))
    new_model.eval()
    print("模型加载成功。")

    torch.jit.load(scripted_path)
    print("TorchScript 加载成功。")
    return model
