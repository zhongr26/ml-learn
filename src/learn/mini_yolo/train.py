from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from config import MINI_YOLO_CFG, settings
from src.learn.mini_yolo.lit_module import LitMiniYOLO
from src.learn.mini_yolo.synth_data import SynthDetDataModule


def train(retrain: bool | None = None) -> Path:
    best = MINI_YOLO_CFG.checkpoint_dir / "best.ckpt"
    if retrain is None:
        retrain = settings.retrain
    if best.exists() and not retrain:
        print(f"[train] 复用已有权重: {best}（设 RETRAIN=1 强制重训）")
        return best

    hp = MINI_YOLO_CFG.hparams
    L.seed_everything(hp.seed)
    dm = SynthDetDataModule(
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
    )

    model = LitMiniYOLO(lr=hp.learning_rate)

    ckpt_callback = callbacks.ModelCheckpoint(
        # 直接存到统一 checkpoint 目录，保证返回的 best.ckpt 与复用逻辑一致
        dirpath=MINI_YOLO_CFG.checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1, save_last=True, verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hp.patience,
    )
    logger = [
        TensorBoardLogger(save_dir=MINI_YOLO_CFG.log_dir, name="mini-yolo"),
        CSVLogger(save_dir=MINI_YOLO_CFG.log_dir, name="mini_yolo_csv")
    ]

    trainer = L.Trainer(
        max_epochs=hp.max_epochs,
        accelerator="auto",
        devices="auto",
        # 32 位精度：MX450 上 16-mixed(AMP) 会出现 loss NaN（与 yolo_learn 同坑）
        precision=32,
        callbacks=[ckpt_callback, early_stop, RichProgressBar(), RichModelSummary()],
        logger=logger,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dm)
    print(f"[train] 训练完成，best: {trainer.checkpoint_callback.best_model_path}")
    return best if best.exists() else Path(trainer.checkpoint_callback.best_model_path)
