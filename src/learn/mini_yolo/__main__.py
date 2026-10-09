from src.learn.mini_yolo.lit_module import LitMiniYOLO
from src.learn.mini_yolo.synth_data import SynthDetDataset
from src.learn.mini_yolo.train import train
from src.learn.mini_yolo.viz import demo_detections


def main():
    best = train()
    model = LitMiniYOLO.load_from_checkpoint(best)
    model.eval()

    ds = SynthDetDataset(train=False, n_per_epoch=4)
    for r in range(10):
        demo_detections(model, ds, n=4, round=r)

    from src.learn.mini_yolo.eval import evaluate_map
    result = evaluate_map(model, SynthDetDataset(train=False), n=500)
    print(f"[main] mAP@0.5 = {result['map50']:.4f}（500 张验证样本）")


if __name__ == '__main__':
    main()
