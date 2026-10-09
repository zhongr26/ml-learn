"""
MINI_YOLO 项目配置：HParams（含项目专属字段）+ Dataset 单例。

环境变量前缀 MINI_YOLO_，读 .env.mini_yolo（项目覆盖，可选）与 .env（全局默认）。
数据复用已下载的 MNIST（合成画布的数字来源），产物目录 subdir=mini_yolo。
"""

from config.base import CommonHParams, Dataset, _dataset_config_cls


class MiniYoloHParams(CommonHParams):
    # ---------- 画布与网格（改动需保持 canvas / grid = 下采样倍数为 2 的幂） ----------
    canvas: int = 64  # 画布边长
    digit_size: int = 32  # 数字缩放后的边长
    grid: int = 4  # 网格数 S，下采样倍数 = canvas // grid = 16
    num_cls: int = 10  # 类别数（数字 0~9）
    max_objs: int = 2  # 每张画布最多贴几个数字
    n_boxes: int = 2  # 每格预测框数 B（论文 YOLOv1 B=2）

    # ---------- 数据规模（每个 epoch 的虚拟样本数） ----------
    n_train: int = 4000
    n_val: int = 500

    # ---------- 损失权重（步骤 4 用） ----------
    lambda_box: float = 5.0  # 有物体格子的框回归权重
    lambda_noobj: float = 0.5  # 无物体格子的 obj 抑制权重


MiniYoloConfig = _dataset_config_cls("mini_yolo", "MINI_YOLO_", MiniYoloHParams)

MINI_YOLO_CFG = Dataset("mini_yolo", "MINI_YOLO_", MiniYoloHParams)
