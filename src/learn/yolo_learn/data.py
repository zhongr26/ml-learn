from ultralytics import settings as yolo_settings

from config.yolo_config import YOLO_CFG


def prepare_data(data_yaml=None):
    """确保 COCO 数据集就位，返回 ultralytics 解析后的数据集 dict。"""
    # ultralytics 全局设置：把数据集根目录指到项目 datasets/<subdir> 下
    # （ultralytics 默认下载到用户目录，这里收编进项目，写一次存 AppData）
    yolo_settings.update({"datasets_dir": str(YOLO_CFG.data_dir)})

    # check_det_dataset：解析 yaml -> 绝对路径 dict；数据缺失时自动触发下载
    from ultralytics.data.utils import check_det_dataset

    yaml_name = data_yaml or YOLO_CFG.hparams.data_yaml  # 默认 coco8.yaml
    return check_det_dataset(yaml_name)


def dataset_info(data_yaml=None) -> dict:
    """返回数据集元信息：train/val 路径、类别名、类别数。"""
    data = prepare_data(data_yaml)
    names = data["names"]  # {0:'person', 1:'bicycle', ...} 共 80 类

    return {
        "yaml": data.get("yaml_file"),  # yaml 文件绝对路径
        "train": data.get("train"),  # 训练集图片目录
        "val": data.get("val"),  # 验证集图片目录
        "nc": data.get("nc", len(names)),  # 类别数（number of classes）
        "names": names,
    }


def sample_images(split="val", n=4):
    """取数据集验证集的 n 张样本图路径，供推理/可视化 demo 使用。"""
    from pathlib import Path

    data = prepare_data()
    img_dir = Path(data["val"])  # 验证集目录
    if not img_dir.is_absolute():  # yaml 里可能是相对路径
        img_dir = (YOLO_CFG.data_dir / img_dir).resolve()
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    return imgs[:n]
