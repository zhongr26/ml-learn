import matplotlib.pyplot as plt

from config import MINI_YOLO_CFG, setup_matplotlib_chinese
from src.learn.mini_yolo.predict import predict_detections
from src.learn.mini_yolo.synth_data import CANVAS


def demo_detections(model, dataset, n=4, tag="detections", round=None):
    setup_matplotlib_chinese()
    out_dir = MINI_YOLO_CFG.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    for ax, i in zip(axes if n > 1 else [axes], range(n)):
        img, labels = dataset[i]
        dets = predict_detections(model, img)
        ax.imshow(img[0], cmap="gray")
        for d in dets:  # 预测框: 红色
            x1, y1, x2, y2 = [v * CANVAS for v in d["xyxy"]]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="red", lw=2))
            ax.text(x1, y1 - 2, f"{d['cls']} {d['conf']:.2f}",
                    color="red", fontsize=9)
        for cx, cy, w, h, cls in labels:  # GT 框: 绿色虚线
            if cls < 0:
                continue
            x, y = (cx - w / 2) * CANVAS, (cy - h / 2) * CANVAS
            ax.add_patch(plt.Rectangle((x, y), w * CANVAS, h * CANVAS,
                                       fill=False, edgecolor="lime",
                                       linestyle="--", lw=1.5))
        ax.set_title(f"{len(dets)} 检出")
        ax.axis("off")
    plt.tight_layout()

    if round is not None:
        out = out_dir / f"detections-{round}.png"
    else:
        out = out_dir / "detections.png"

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[viz] {out}")
