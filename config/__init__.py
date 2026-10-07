"""
配置包总出口。

用法：
    from config import settings, MNIST, YOLO
    settings.retrain                  # 全局开关
    MNIST.data_dir / YOLO.output_dir  # 项目目录
    MNIST.hparams / YOLO.hparams      # 项目有效超参（全局 + 项目覆盖）

调试: python -m config
"""
from config.base import PROJECT_ROOT, Dataset, setup_matplotlib_chinese
from config.global_config import settings
from config.mnist_config import MNIST
from config.yolo_config import YOLO

__all__ = [
    "PROJECT_ROOT",
    "settings",
    "MNIST",
    "YOLO",
    "Dataset",
    "setup_matplotlib_chinese",
]
