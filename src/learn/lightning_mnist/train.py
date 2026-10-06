import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import (SEED, MNIST_CHECKPOINT_DIR, SAVE_TOP_K, PATIENCE, MNIST_LOG_DIR, MAX_EPOCHS, LATENT_DIM,
                    LEARNING_RATE, RETRAIN)
from .data import MNISTDataModule
from .model import LitAutoModel


def train():
    ckpt_dir = MNIST_CHECKPOINT_DIR / "mnist"
    last_ckpt = ckpt_dir / "last.ckpt"

    # 已有训练好的 checkpoint 时跳过训练，直接复用（RETRAIN=1 强制重新训练）
    if not RETRAIN and last_ckpt.exists():
        print(f"[train] 检测到已有 checkpoint，跳过训练: {last_ckpt}")
        print("[train] 如需重新训练，请设置环境变量 RETRAIN=1")
        return str(last_ckpt)

    L.seed_everything(SEED, workers=True)
    dm = MNISTDataModule()

    ckpt_callback = callbacks.ModelCheckpoint(
        dirpath=MNIST_CHECKPOINT_DIR / "mnist",
        filename="mnist-ae-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min",
        save_top_k=SAVE_TOP_K, save_last=True, verbose=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    logger = TensorBoardLogger(save_dir=MNIST_LOG_DIR, name="mnist_autoencoder")

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[ckpt_callback, early_stop],
        logger=logger,
        log_every_n_steps=50,
    )

    model = LitAutoModel(latent_dim=LATENT_DIM, lr=LEARNING_RATE)
    trainer.fit(model, datamodule=dm)
    print(f"\n最佳模型路径: {ckpt_callback.best_model_path}")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    return ckpt_callback.best_model_path
