"""
配置包总出口。

用法：
    from config import settings, MNIST_CFG, YOLO_CFG
    settings.retrain                  # 全局开关
    MNIST_CFG.data_dir / YOLO_CFG.output_dir  # 项目目录
    MNIST_CFG.hparams / YOLO_CFG.hparams      # 项目有效超参（全局 + 项目覆盖）

调试: python -m config
"""
from config.base import PROJECT_ROOT, Dataset, setup_matplotlib_chinese
from config.global_config import settings
from config.mini_yolo_config import MINI_YOLO_CFG
from config.mnist_config import MNIST_CFG
from config.yolo_config import YOLO_CFG

__all__ = [
    "PROJECT_ROOT",
    "settings",
    "MNIST_CFG",
    "YOLO_CFG",
    "MINI_YOLO_CFG",
    "Dataset",
    "setup_matplotlib_chinese",
]
