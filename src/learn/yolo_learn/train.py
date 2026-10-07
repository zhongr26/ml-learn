"""
训练入口。

与 lightning_mnist.train 保持同样的重训策略：
- 已有训练产物（best.pt）且未开 RETRAIN 时直接复用，跳过训练
- 训练产物放在 YOLO_CFG.log_dir / YOLO_CFG.checkpoint_dir 下

ultralytics 训练:
    model.train(data=..., epochs=..., imgsz=..., batch=..., project=..., name=...)
产物为 project/name/weights/{best.pt,last.pt} 及 results.png 等日志图。
"""

from pathlib import Path

import torch

from config import YOLO_CFG, settings
from src.learn.yolo_learn.data import prepare_data
from src.learn.yolo_learn.model import resolve_weights


def _run_dir() -> Path:
    """训练运行目录：logs/yolo/train/（ultralytics 会在其下建 exp/）。"""
    d = YOLO_CFG.log_dir / "train"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train(retrain: bool | None = None) -> Path:
    """训练（或复用）模型，返回 best 权重路径。"""
    from ultralytics import YOLO

    best = YOLO_CFG.checkpoint_dir / "best.pt"
    if retrain is None:
        retrain = settings.retrain
    if best.exists() and not retrain:
        print(f"[train] 复用已有权重: {best}（设 RETRAIN=1 强制重训）")
        return best

    hp = YOLO_CFG.hparams
    data = prepare_data(hp.data_yaml)
    run_dir = _run_dir()

    model = YOLO(resolve_weights(hp.yolo_model))  # 权重统一放 checkpoints/yolo/
    model.train(
        data=data.get("yaml_file") or hp.data_yaml,
        epochs=hp.max_epochs,
        imgsz=hp.imgsz,
        # batch_size <= 0（或 .env.yolo 中 YOLO_BATCH_SIZE=-1）视为未指定 -> autobatch
        batch=hp.batch_size if hp.batch_size > 0 else -1,
        workers=hp.num_workers,
        seed=hp.seed,
        patience=hp.patience,  # 早停：验证指标 n 轮不提升就停
        amp=hp.amp,  # 混合精度开关（见 yolo_config 注释）
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(run_dir),  # 运行根目录
        name="exp",  # 本次运行子目录
        exist_ok=True,  # 目录已存在时不改名 exp2/exp3...
    )

    # ultralytics 把权重存在 project/name/weights/ 下，拷贝到统一 checkpoint 目录
    best_src = Path(model.trainer.best)
    YOLO_CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best.write_bytes(best_src.read_bytes())
    print(f"[train] 训练完成，best 权重: {best}")
    return best
