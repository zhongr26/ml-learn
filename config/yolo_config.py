"""
YOLO 项目配置：HParams（含项目专属字段）+ Dataset 单例。

环境变量前缀 YOLO_，读 .env.yolo（项目覆盖，可选）与 .env（全局默认）。
"""
from pydantic import BaseModel

from config.base import CommonHParams, Dataset, _dataset_config_cls


class YoloHParams(CommonHParams):
    yolo_model: str = "yolo11n.pt"  # 预训练权重，n/s/m/l/x 规模递增
    data_yaml: str = "coco8.yaml"  # ultralytics 数据集定义（首次训练自动下载）
    conf: float = 0.25  # 推理置信度阈值
    iou: float = 0.7  # NMS IoU 阈值
    imgsz: int = 640  # 推理/训练输入尺寸


YoloConfig = _dataset_config_cls("yolo", "YOLO_", YoloHParams)

YOLO = Dataset("yolo", "YOLO_", YoloHParams)
