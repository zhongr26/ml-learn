import logging
import warnings
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks import EarlyStopping, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from config import MNIST, settings
from .data import MNISTDataModule
from .model import LitAutoModel


class _NoTips(logging.Filter):
    """过滤 litlogger 推广提示等噪声日志。"""
    def filter(self, record):
        return "litlogger" not in record.getMessage()


def _setup_logging():
    log_file = MNIST.log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    no_tips = _NoTips()
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    for h in handlers:
        h.addFilter(no_tips)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # Lightning 内部告警（非代码问题）：混合精度下模型大小估算不准 / lightning 加载 ckpt 的 torch.load 提示
    warnings.filterwarnings("ignore", message=".*16-mixed.*model summary.*")
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    return log_file


def train():
    hp = MNIST.hparams
    log_file = _setup_logging()
    print(f"[train] 日志文件: {log_file}")
    ckpt_dir = MNIST.checkpoint_dir / "mnist"
    last_ckpt = ckpt_dir / "last.ckpt"

    # 已有训练好的 checkpoint 时跳过训练，直接复用（.env 中 RETRAIN=1 强制重新训练）
    if not settings.retrain and last_ckpt.exists():
        print(f"[train] 检测到已有 checkpoint，跳过训练: {last_ckpt}")
        print("[train] 如需重新训练，请设置环境变量 RETRAIN=1")
        return str(last_ckpt)

    L.seed_everything(hp.seed, workers=True)
    dm = MNISTDataModule()

    ckpt_callback = callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="mnist-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=hp.save_top_k, save_last=True, verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=hp.patience,
    )
    logger = [TensorBoardLogger(save_dir=MNIST.log_dir, name="mnist_autoencoder"),
              CSVLogger(save_dir=MNIST.log_dir, name="mnist_autoencoder_csv")]

    trainer = L.Trainer(
        max_epochs=hp.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[ckpt_callback, early_stop, RichProgressBar(), RichModelSummary()],
        logger=logger,
        log_every_n_steps=50,
    )

    model = LitAutoModel(
        latent_dim=hp.latent_dim,
        lr=hp.learning_rate,
        ssim_weight=hp.ssim_weight,
        cls_weight=hp.cls_weight,
    )
    trainer.fit(model, datamodule=dm)
    print(f"\n最佳模型路径: {ckpt_callback.best_model_path}")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    return ckpt_callback.best_model_path
