"""
MNIST_CFG 项目配置：HParams（含项目专属字段）+ Dataset 单例。

环境变量前缀 MNIST_，读 .env.mnist（项目覆盖，可选）与 .env（全局默认）。
"""

from config.base import CommonHParams, Dataset, _dataset_config_cls


class MnistHParams(CommonHParams):
    latent_dim: int = 3
    ssim_weight: float = 0.5
    cls_weight: float = 0.5


MnistConfig = _dataset_config_cls("mnist", "MNIST_", MnistHParams)

MNIST_CFG = Dataset("mnist", "MNIST_", MnistHParams)
