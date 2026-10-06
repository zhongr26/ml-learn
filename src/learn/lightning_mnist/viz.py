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


@torch.no_grad()
def demo_tsne_umap(model, dm, max_samples=3000, method="tsne"):
    """t-SNE 或 UMAP 可视化潜在空间。"""
    model.eval()
    device = next(model.parameters()).device

    # 收集潜在向量
    zs, ys = [], []
    total = 0
    for x, y in dm.test_dataloader():
        z = model(x.to(device)).cpu()
        zs.append(z);
        ys.append(y)
        total += x.size(0)
        if total >= max_samples:
            break
    z = torch.cat(zs).numpy()[:max_samples]
    y = torch.cat(ys).numpy()[:max_samples]

    # 降维
    if method == "tsne":
        from sklearn.manifold import TSNE
        z_2d = TSNE(
            n_components=2, perplexity=30,
            init="pca", learning_rate="auto",
            random_state=42, n_jobs=-1,
        ).fit_transform(z)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("请先安装: pip install umap-learn")
        z_2d = umap.UMAP(
            n_components=2, n_neighbors=15,
            min_dist=0.1, random_state=42,
        ).fit_transform(z)
    else:
        raise ValueError(f"未知方法: {method}")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap="tab10",
                    s=5, alpha=0.7)
    plt.colorbar(sc, label="数字标签", ticks=range(10))
    ax.set_title(f"MNIST 潜在空间 ({method.upper()}, n={max_samples})")
    ax.set_xticks([]);
    ax.set_yticks([])

    save_fig = MNIST_OUTPUT_DIR / f"latent_{method}.png"
    plt.tight_layout()
    plt.savefig(save_fig, dpi=120)
    plt.show()
    print(f"{method.upper()} 可视化已保存: {save_fig}")


@torch.no_grad()
def demo_interpolation(model, dm, n_steps=10):
    """取两张不同数字的图，在潜在空间线性插值，看解码渐变。"""
    model.eval()
    device = next(model.parameters()).device

    x, y = next(iter(dm.test_dataloader()))
    x = x.to(device)
    x_flat = x.view(x.size(0), -1)

    z1 = model(x_flat[:1])  # 第 1 张
    z2 = model(x_flat[1:2])  # 第 2 张

    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
    z_interp = z1 * (1 - alphas) + z2 * alphas  # (n_steps, latent_dim)
    x_interp = model.decoder(z_interp).view(-1, 1, 28, 28)

    grid = make_grid(x_interp, nrow=n_steps, normalize=True)
    save_image(grid, MNIST_OUTPUT_DIR / "interpolation.png")
    print(f"插值图已保存 (数字 {y[0].item()} → {y[1].item()})")


@torch.no_grad()
def demo_traversal(model, dm, dim=0, n_steps=10, span=3.0):
    """遍历潜在空间第 dim 维，观察其语义。"""
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.hparams.latent_dim

    z = torch.zeros(1, latent_dim, device=device)
    vals = torch.linspace(-span, span, n_steps)
    zs = z.repeat(n_steps, 1)
    zs[:, dim] = vals

    x = model.decoder(zs).view(-1, 1, 28, 28)
    grid = make_grid(x, nrow=n_steps, normalize=True)
    save_image(grid, MNIST_OUTPUT_DIR / f"traversal_dim{dim}.png")
    print(f"维度 {dim} 遍历图已保存")


@torch.no_grad()
def demo_diff_heatmap(model, dm, n=8):
    model.eval()
    device = next(model.parameters()).device
    x, _ = next(iter(dm.test_dataloader()))
    x = x.to(device)
    x_flat = x.view(x.size(0), -1)
    x_hat = model.decoder(model(x)).view(-1, 1, 28, 28)

    diff = (x[:n] - x_hat[:n]).abs()  # (n,1,28,28)
    grid = make_grid(diff, nrow=n, normalize=True)
    save_image(grid, MNIST_OUTPUT_DIR / "diff_heatmap.png")


@torch.no_grad()
def demo_per_class(model, dm, per_class=8):
    model.eval()
    device = next(model.parameters()).device
    x, y = next(iter(dm.test_dataloader()))
    x, y = x.to(device), y.to(device)
    x_flat = x.view(x.size(0), -1)
    x_hat = model.decoder(model(x)).view(-1, 1, 28, 28)

    rows = []
    for c in range(10):
        idx = (y == c).nonzero(as_tuple=True)[0][:per_class]
        if len(idx) == 0:
            continue
        rows.append(x[idx])
        rows.append(x_hat[idx])
    grid = make_grid(torch.cat(rows), nrow=per_class, normalize=True)
    save_image(grid, MNIST_OUTPUT_DIR / "per_class_grid.png")
