"""
可视化 demo，产物保存到 outputs/yolo/。

- demo_inference: 对样本图推理，保存带检测框的结果图 + 检测清单
- demo_class_distribution: 统计一批图中各类别出现次数并画条形图
"""
from collections import Counter

import matplotlib.pyplot as plt

from config import YOLO_CFG, setup_matplotlib_chinese
from src.learn.yolo_learn.model import YOLOModel


def demo_inference(model: YOLOModel, image_paths, tag="inference"):
    """对样本图逐张推理，保存带框结果图到 output_dir，打印检测清单。"""
    out_dir = YOLO_CFG.output_dir / tag  # outputs/yolo/inference/
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        results = model.predict(str(img_path))
        dets = model.boxes_to_list(results[0])  # 结构化检测列表
        print(f"[viz] {img_path.name}: {len(dets)} 个目标")
        for d in dets:
            print(f"    {d['name']:>12} {d['conf']:.2f}  "
                  f"({d['xyxy'][0]:.0f},{d['xyxy'][1]:.0f})-({d['xyxy'][2]:.0f},{d['xyxy'][3]:.0f})")

        saved = out_dir / img_path.name
        results[0].save(filename=str(saved))  # 保存带框图（ultralytics 自动画框）
        print(f"[viz] 结果图: {saved}")


def demo_class_distribution(model: YOLOModel, image_paths, tag="class_dist"):
    """统计样本图上的类别分布并保存条形图。"""
    counter = Counter()  # 类别名 -> 出现次数
    for img_path in image_paths:
        results = model.predict(source=str(img_path))
        counter.update(d["name"] for d in model.boxes_to_list(results[0]))

    if not counter:
        print("[viz] 无检测结果")
        return

    setup_matplotlib_chinese()  # 中文标签必须先配字体
    names, counts = zip(*counter.most_common())  # 按次数降序
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, counts, color="steelblue")
    ax.set_title("COCO 样本检测结果类别分布")
    ax.set_ylabel("出现次数")
    plt.xticks(rotation=45, ha="right")  # 类别名多时斜着放
    fig.tight_layout()

    out = YOLO_CFG.output_dir / f"{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[viz] 类别分布图: {out}")
