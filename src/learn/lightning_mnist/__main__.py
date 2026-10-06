from src.learn.lightning_mnist.io_utils import save_and_load_demo
from src.learn.lightning_mnist.train import train
from src.learn.lightning_mnist.viz import demo_reconstruction, demo_latent_space


def main():
    best_ckpt = train()
    model = save_and_load_demo(best_ckpt)
    demo_reconstruction(model)
    demo_latent_space(model)


if __name__ == "__main__":
    main()
