from src.learn.lightning_mnist.data import MNISTDataModule
from src.learn.lightning_mnist.io_utils import save_and_load_demo
from src.learn.lightning_mnist.train import train
from src.learn.lightning_mnist.viz import (
    demo_reconstruction,
    demo_diff_heatmap,
    demo_per_class,
    demo_interpolation,
    demo_traversal,
    demo_tsne_umap,
)


def main():
    best_ckpt = train()
    model = save_and_load_demo(best_ckpt)
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup("test")

    # 基础重建
    demo_reconstruction(model)
    demo_diff_heatmap(model, dm)
    demo_per_class(model, dm)

    # 潜在空间探索
    demo_interpolation(model, dm)
    for d in range(min(3, model.hparams.latent_dim)):
        demo_traversal(model, dm, dim=d)

    # 高维潜在空间降维可视化
    demo_tsne_umap(model, dm, max_samples=3000, method="tsne")
    demo_tsne_umap(model, dm, max_samples=3000, method="umap")


if __name__ == "__main__":
    main()
