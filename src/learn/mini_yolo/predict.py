import torch

from src.learn.mini_yolo.lit_module import LitMiniYOLO
from src.learn.mini_yolo.loss import pairwise_iou, cxcywh_to_xyxy
from src.learn.mini_yolo.targets import decode


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.4):
    """(N,4) xyxy 贪心 NMS，返回保留索引。与 core.py 相同，此处自含。"""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if rest.numel() == 0:
            break
        iou = pairwise_iou(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou <= iou_thr]
    return keep


@torch.no_grad()
def predict_detections(model: LitMiniYOLO, img, conf_thr=0.25, iou_thr=0.4):
    """
    model: LitMiniYOLO；img: (1,H,W) 归一化画布。
    返回 list of dict: {xyxy, conf, cls}（坐标为画布归一化值）。
    """
    model.eval()
    img = img.to(model.device)  # 训练在 GPU 时模型权重在 cuda，输入需同设备
    pred = model(img.unsqueeze(0))[0]  # (S, S, 5B+C) 已 sigmoid
    raw = decode(pred.unsqueeze(0), grid=model.grid, n_boxes=model.n_boxes)[0]  # (S, S, B, 6)
    p = raw.reshape(-1, 6)  # (N, 4) 展平所有候选框

    # 1) 置信度过滤
    p = p[p[:, 4] > conf_thr]
    if len(p) == 0:
        return []

    # 2) NMS 去除冗余
    xyxy = cxcywh_to_xyxy(p[:, :4])
    keep = nms(xyxy, p[:, 4], iou_thr=iou_thr)
    return [{
        "xyxy": xyxy[k].tolist(),
        "conf": float(p[k, 4]),
        "cls": int(p[k, 5]),
    } for k in keep]
