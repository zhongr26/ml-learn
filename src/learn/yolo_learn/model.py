"""
YOLO_CFG 模型封装。

对 ultralytics.YOLO 做一层薄包装：
- 统一从 config 读取超参（权重、conf、iou、imgsz、device）
- 权重统一存放在 checkpoints/yolo/，缺失时自动下载到那里（不在 cwd 散落）
- 提供 predict 封装与结果转结构的工具方法

注意：ultralytics 自带训练/推理循环，这里不做成 LightningModule；
后续阶段（手写组件/迷你 YOLO）再引入 Lightning 化的训练。
"""
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from config import YOLO_CFG


def auto_device():
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_weights(weights) -> str:
    """把权重名（如 yolo11n.pt）解析为 checkpoints/yolo/ 下的路径，缺失则下载到那里。"""
    p = Path(weights)
    if p.exists():  # 已是有效路径（绝对路径或相对 cwd）
        return str(p)
    target = YOLO_CFG.checkpoint_dir / p.name
    if not target.exists():
        attempt_download_asset(str(target))  # ultralytics 官方下载，直接落到 target
    return str(target)


class YOLOModel:
    """YOLO 检测模型：默认加载 config 指定的预训练权重。"""

    def __init__(self, weights=None):
        hp = YOLO_CFG.hparams
        self.weights = resolve_weights(weights if weights else hp.yolo_model)
        self.model = YOLO(self.weights)
        self.names = self.model.names

    def predict(self, source, **overrides):
        """推理，返回 Results 列表。超参默认来自 config，可逐项覆盖；不负责存盘（由 viz 负责）。"""
        hp = YOLO_CFG.hparams
        kwargs = dict(
            conf=hp.conf,
            iou=hp.iou,
            imgsz=hp.imgsz,
            device=auto_device(),
            verbose=False,  # 关掉每张图的日志刷屏
        )
        kwargs.update(overrides)
        return self.model.predict(source, **kwargs)

    @staticmethod
    def boxes_to_list(r):
        """Results -> [{cls, name, conf, xyxy}]，便于打印与后续分析。"""
        out = []
        for box in r.boxes:  # r.boxes 是所有检测框的张量集合
            out.append({
                "cls": int(box.cls),  # 类别索引（张量 -> int）
                "name": r.names[int(box.cls)],  # 类别名
                "conf": float(box.conf),  # 置信度 0~1
                "xyxy": box.xyxy[0].tolist(),  # [x1,y1,x2,y2] 左上/右下角点（像素）
            })
        return out
