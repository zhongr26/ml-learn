"""
权重导出与重载 demo。

ultralytics 支持导出 ONNX / TorchScript / TensorRT 等格式：
    model.export(format="torchscript")  -> 导出文件路径
导出后的文件同样用 YOLO(path) 加载，接口与 .pt 完全一致。
"""
from ultralytics import YOLO

from config import YOLO_CFG


def save_and_load_demo(weights) -> str:
    """导出 TorchScript 并重载验证，返回导出文件路径。"""
    export_path = YOLO(weights).export(format="torchscript", imgsz=YOLO_CFG.hparams.imgsz)
    print(f"[io] 导出 TorchScript: {export_path}")

    loaded = YOLO(export_path)  # 用导出文件重新加载
    names = loaded.names
    print(f"[io] 重载成功，类别数: {len(names)}（{next(iter(names.values()))} 等）")
    return export_path
