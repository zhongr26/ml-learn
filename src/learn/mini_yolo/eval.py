"""
模型评价指标：mAP@0.5（对齐阶段 3 手写的 AP 逻辑）。

评估流程（按类聚合全部图片的结果）:
1. 逐图推理 -> 置信度过滤 -> NMS 得到候选框
2. 每类内按置信度降序判定 TP/FP（IoU>=thr 且 GT 未被占用）
3. 前缀和 -> PR 曲线 -> precision 包络 -> 积分得 AP
4. 有 GT 的类别取平均 = mAP

运行: 训练完成后 python -m src.learn.mini_yolo.eval
"""
from collections import defaultdict

import torch

from src.learn.mini_yolo.loss import cxcywh_to_xyxy, pairwise_iou
from src.learn.mini_yolo.predict import predict_detections


@torch.no_grad()
def evaluate_map(model, dataset, n=500, iou_thr=0.5, conf_thr=0.1) -> dict:
    """在合成数据集前 n 张上评估 mAP@0.5，返回 {map50, ap_per_class}。"""
    # cls -> [(conf, is_tp), ...] 按图序收集（同图内稍后排序）
    records = defaultdict(list)
    n_gt = defaultdict(int)

    for i in range(min(n, len(dataset))):
        img, labels = dataset[i]
        preds = predict_detections(model, img, conf_thr=conf_thr)

        gts = [g for g in labels if g[4] >= 0]
        for g in gts:
            n_gt[int(g[4])] += 1

        gt_xyxy = cxcywh_to_xyxy(torch.stack([g[:4] for g in gts])) if gts else None
        matched = set()  # 已被占用的 GT 索引

        # 同类内按置信度降序判定
        preds_sorted = sorted(preds, key=lambda d: d["conf"], reverse=True)
        for d in preds_sorted:
            cls = d["cls"]
            p = torch.tensor(d["xyxy"]).unsqueeze(0)
            is_tp = False
            if gt_xyxy is not None:
                same_cls = [k for k, g in enumerate(gts) if int(g[4]) == cls and k not in matched]
                if same_cls:
                    ious = pairwise_iou(p, gt_xyxy[same_cls])[0]
                    best_iou, best = ious.max(0)
                    if best_iou >= iou_thr:
                        matched.add(same_cls[best.item()])
                        is_tp = True
            records[cls].append((d["conf"], is_tp))

    # 每类 PR -> AP
    ap_per_class = {}
    for cls, recs in records.items():
        if n_gt[cls] == 0:
            continue
        recs.sort(key=lambda r: r[0], reverse=True)
        tp = torch.tensor([1.0 if t else 0.0 for _, t in recs])
        tp_cum = tp.cumsum(0)
        fp_cum = (1 - tp).cumsum(0)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall = tp_cum / n_gt[cls]
        # precision 包络（从后往前取右侧最大值）
        for k in range(len(precision) - 2, -1, -1):
            precision[k] = torch.max(precision[k], precision[k + 1])
        recall = torch.cat([recall, torch.tensor([1.0])])
        precision = torch.cat([precision, torch.tensor([0.0])])
        ap_per_class[cls] = ((recall[1:] - recall[:-1]) * precision[1:]).sum().item()

    map50 = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    return {"map50": map50, "ap_per_class": ap_per_class, "n_gt": dict(n_gt)}


if __name__ == "__main__":
    from config import MINI_YOLO_CFG
    from src.learn.mini_yolo.lit_module import LitMiniYOLO
    from src.learn.mini_yolo.synth_data import SynthDetDataset

    best = MINI_YOLO_CFG.checkpoint_dir / "best.ckpt"
    model = LitMiniYOLO.load_from_checkpoint(best)
    model.eval()
    ds = SynthDetDataset(train=False)
    result = evaluate_map(model, ds, n=500)
    print(f"mAP@0.5 = {result['map50']:.4f}  (500 张验证样本)")
    top = sorted(result["ap_per_class"].items(), key=lambda kv: -kv[1])[:5]
    print("AP 最高 5 类:", {k: round(v, 3) for k, v in top})
    worst = sorted(result["ap_per_class"].items(), key=lambda kv: kv[1])[:5]
    print("AP 最低 5 类:", {k: round(v, 3) for k, v in worst})
