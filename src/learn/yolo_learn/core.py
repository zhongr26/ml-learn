import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
   计算两组框的两两 IoU 矩阵。

   参数:
       boxes1: (N, 4) xyxy 格式（左上右下）
       boxes2: (M, 4) xyxy 格式
   返回:
       iou: (N, M)，iou[i, j] = boxes1[i] 与 boxes2[j] 的交并比

   要点: 全程用广播，不写双重循环。
   交集宽高用 clamp_min(0)：框不相交时负数要截成 0。
   """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 交集，左上取较大，右下取较小 -> (N, M, 2)
    # max( (N, 1, 2), (1, M, 2) ) 用广播机制得到 (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    # 取第一或最后一个维度，等价于 wh[:, :, 0]，忽略维度的数量
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """
    贪心 NMS：每轮取最高分框保留，删除与它 IoU 超阈值的框。

    参数:
        boxes:  (N, 4) xyxy，同一类别的框
        scores: (N,) 置信度
    返回:
        keep: 保留下框的索引（按分数降序）

    要点: 为什么删 IoU 大的而不是留分数接近的 ——
    同一物体上多个框高度重叠，重叠即冗余；不同物体即使分数相同也不重叠。
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)  # 按分数索引

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        # 与其余框的 iou
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CIoU loss = 1 - CIoU，其中
    CIoU = IoU - (中心点距离惩罚 + 长宽比惩罚)

    参数: pred/target 均为 (N, 4) xyxy
    返回: 标量 loss（可反传）

    对比 L1/IoU:
    - L1 对每个坐标独立，框大框小惩罚一样，且与 IoU 指标不相关
    - IoU loss 与指标直接相关但：不相交时 IoU=0 梯度消失
    - CIoU 额外惩罚中心偏移和长宽比，不相交时也有梯度
    """
    iou = box_iou(pred, target).diagonal()  # (N, )

    # 中心点距离惩罚
    # 最小包围框（能同时罩住两框的框）对角线长度
    enclose_lt = torch.min(pred[:, :2], target[:, :2])
    enclose_rb = torch.max(pred[:, 2:], target[:, 2:])
    c2 = ((enclose_rb - enclose_lt) ** 2).sum(dim=1).clamp(min=1e-9)

    # 两框中心点距离的平方
    pred_c = (pred[:, :2] + pred[:, 2:]) / 2
    target_c = (target[:, :2] + target[:, 2:]) / 2
    d2 = ((pred_c - target_c) ** 2).sum(dim=1)

    # 长宽比惩罚
    import math
    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=0)
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_w = (target[:, 2] - target[:, 0]).clamp(min=0)
    target_h = (target[:, 3] - target[:, 1]).clamp(min=0)

    v = (4 / math.pi ** 2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    alpha = v / (1 - iou + v).clamp(min=1e-9)

    ciou = iou - d2 / c2 - alpha * v
    return (1 - ciou).mean()


def ap_at_iou(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """
    mAP评估：单类别单张图集合的 AP（area under PR curve, 11点插值简化为精确积分）。

    参数（整批数据拍平后传入）:
        pred_boxes:  (P, 4) 全部预测框
        pred_scores: (P,)
        gt_boxes:    (G, 4) 全部真值框
    返回:
        ap: 该 IoU 阈值下的平均精度

    核心概念:
    - TP: 预测框与某个未匹配过的真值 IoU >= thr（每个真值只能被匹配一次）
    - FP: 没匹配上任何真值的预测（重复检测、背景误检都算 FP）
    - 按分数降序逐个判定 -> 累积计算 precision/recal -> PR 曲线下面积
    """
    if len(gt_boxes) == 0:
        return float("nan")

        # 按分数降序处理预测
    order = pred_scores.argsort(descending=True)
    matched = torch.zeros(len(gt_boxes), dtype=torch.bool)  # 真值是否已被匹配
    tp = torch.zeros(len(order))
    fp = torch.zeros(len(order))

    ious = box_iou(pred_boxes[order], gt_boxes)  # (P, G)

    for rank in range(len(order)):
        iou_row = ious[rank]
        best_iou, best_gt = iou_row.max(dim=0) if len(gt_boxes) else (torch.tensor(0.), 0)

        if best_iou >= iou_thr and not matched[best_gt]:
            tp[rank] = 1
            matched[best_gt] = True  # 关键：一个真值只匹配一次，第二次算 FP
        else:
            fp[rank] = 1

    # 累积 TP/FP -> PR 曲线
    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / len(gt_boxes)

    # PR 曲线下面积（在 recall 召回率增加处取右侧最大 precision，即 envelope）
    recall = torch.cat([recall, torch.tensor([1.0])])
    precision = torch.cat([precision, torch.tensor([0.0])])

    # PR 曲线下面积（在 recall 召回率增加处取右侧最大 precision，即 envelope）
    recall = torch.cat([recall, torch.tensor([1.0])])
    precision = torch.cat([precision, torch.tensor([0.0])])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = torch.max(precision[i], precision[i + 1])

    # recall 变化的区间上积分 precision
    dr = recall[1:] - recall[:-1]
    return (dr * precision[1:]).sum().item()


# test
if __name__ == "__main__":
    # IoU: 完全重合=1，不接触=0，半重叠=1/3
    a = torch.tensor([[0., 0., 10., 10.]])
    assert box_iou(a, a).item() == 1.0
    assert box_iou(a, torch.tensor([[20., 20., 30., 30.]])).item() == 0.0
    assert abs(box_iou(a, torch.tensor([[5., 0., 15., 10.]])).item() - 1 / 3) < 1e-6

    # NMS: 两个高度重叠的框只留最高分的
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]

    # CIoU: 完全重合 loss=0；随机框 loss>0 且可反传
    t = torch.tensor([[0., 0., 10., 10.]])
    assert ciou_loss(t, t).item() < 1e-6
    p = torch.tensor([[1., 1., 12., 9.]], requires_grad=True)
    loss = ciou_loss(p, t)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()

    # mAP: 完美预测=1，全部漏检=0
    gt = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
    assert ap_at_iou(gt, torch.tensor([0.9, 0.8]), gt) == 1.0
    assert ap_at_iou(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), gt) == 0.5

    print("all tests passed")
