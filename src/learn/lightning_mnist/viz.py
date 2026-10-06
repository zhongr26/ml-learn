import torch
import torch.utils.data as data
import torchvision as tv
from torchvision.utils import make_grid, save_image

from config import MNIST_DATA_DIR, MNIST_OUTPUT_DIR
from config import setup_matplotlib_chinese

setup_matplotlib_chinese()
import matplotlib.pyplot as plt


@torch.no_grad()
def demo_reconstruction(model, n=8):
    model.eval()
    device = next(model.parameters()).device

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = tv.datasets.MNIST(str(MNIST_DATA_DIR), train=False, transform=transform)
    loader = data.DataLoader(test_set, batch_size=n, shuffle=True)
    x, y = next(iter(loader))
    x = x.to(device)

    z = model(x)
    x_hat = model.decoder(z).view(-1, 1, 28, 28)

    mean, std = 0.1307, 0.3081
    x_vis = x * std + mean
    x_hat_vis = x_hat * std + mean

    save_fig = MNIST_OUTPUT_DIR / "recon_demo.png"
    grid = make_grid(torch.cat([x_vis.cpu(), x_hat_vis.cpu()], dim=0),
                     nrow=n, normalize=False)
    save_image(grid, save_fig)
    print(f"重建对比图已保存: {save_fig}")

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x_vis[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title(f"原图 {y[i].item()}")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat_vis[i, 0].cpu(), cmap="gray")
        axes[1, i].set_title("重建")
        axes[1, i].axis("off")
    plt.tight_layout()
    save_grid_fig = MNIST_OUTPUT_DIR / "recon_grid.png"
    plt.savefig(save_grid_fig, dpi=120)
    plt.show()
    print(f"matplotlib 对比图已保存: {save_grid_fig}")


@torch.no_grad()
def demo_latent_space(model):
    model.eval()
    device = next(model.parameters()).device

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = tv.datasets.MNIST(str(MNIST_DATA_DIR), train=False, transform=transform)
    loader = data.DataLoader(test_set, batch_size=512, shuffle=False)

    zs, ys = [], []
    for x, y in loader:
        z = model(x.to(device)).cpu()
        zs.append(z)
        ys.append(y)
    z = torch.cat(zs).numpy()
    y = torch.cat(ys).numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=y, cmap="tab10", s=5)
    plt.colorbar(sc, label="digit")
    ax.set_title("MNIST 3D 潜在空间")
    save_fig = MNIST_OUTPUT_DIR / "latent_space.png"
    plt.savefig(save_fig, dpi=120)
    plt.show()
    print(f"潜在空间图已保存: {save_fig}")
