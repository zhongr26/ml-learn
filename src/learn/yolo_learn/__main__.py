from src.learn.yolo_learn.data import dataset_info, sample_images
from src.learn.yolo_learn.io_utils import save_and_load_demo
from src.learn.yolo_learn.model import YOLOModel
from src.learn.yolo_learn.train import train
from src.learn.yolo_learn.viz import demo_inference, demo_class_distribution


def main():
    best = train(retrain=True)  # 已有 best.pt 且未 RETRAIN 时直接复用
    model = YOLOModel(best)  # 用训练好的权重包一层

    info = dataset_info()
    print(f"[main] 数据集: nc={info['nc']}, val={info['val']}")
    print(f"[main] 权重: {model.weights}, 类别数: {len(model.names)}")

    imgs = sample_images(n=4)  # 验证集取 4 张样本
    demo_inference(model, imgs, tag="inference")
    demo_class_distribution(model, imgs, tag="class_dist")

    save_and_load_demo(best)


if __name__ == "__main__":
    main()
