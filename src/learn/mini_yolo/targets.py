"""
目标编码（GT -> 训练 target）与解码（网络 raw 输出 -> 框）。

坐标约定（全项目统一，务必背下来）:
- GT 标签:   (cx, cy, w, h) 相对整张画布归一化到 0~1
- 训练 target: 每个网格单元每框槽位存 5 维（v1 风格，B 个槽位填同值）
    t_x, t_y = cx*S - j, cy*S - i   中心点相对"本格左上角"的偏移 (0~1)
    t_w, t_h = w, h                 宽高直接回归（网络 sigmoid 输出 0~1）
    obj = 1
- 网络输出 (B,S,S,5*n_boxes+C) 已过 sigmoid，与 target 同语义，见 decode()
"""
import torch

from config import MINI_YOLO_CFG

hp = MINI_YOLO_CFG.hparams


def encode_labels(labels: torch.Tensor, grid: int, num_cls: int, n_boxes=None) -> torch.Tensor:
    """
    一个 batch 的 GT 标签 -> 训练用 target。

    参数:
        labels: (B, max_objs, 5) [cx, cy, w, h, cls]，填充行 cls=-1
        grid:   S
        num_cls: C
    返回:
        target: (B, S, S, 5+C)——与网络 raw 输出同形状，但已过逆激活编码，
                无物体格子的 obj=0，cls 全 0
    """
    n_boxes = n_boxes or hp.n_boxes
    B = labels.shape[0]
    target = torch.zeros(B, grid, grid, 5 * n_boxes + num_cls)
    for b in range(B):
        for cx, cy, w, h, cls in labels[b]:
            if cls < 0:
                continue
            j = min(int(cx * grid), grid - 1)
            i = min(int(cy * grid), grid - 1)
            for k in range(n_boxes):  # 两个槽位都填，择优是损失的事
                base = 5 * k
                target[b, i, j, base + 0] = cx * grid - j  # 格内偏移
                target[b, i, j, base + 1] = cy * grid - i
                target[b, i, j, base + 2] = w  # 直接回归
                target[b, i, j, base + 3] = h
                target[b, i, j, base + 4] = 1.0
            target[b, i, j, 5 * n_boxes + int(cls)] = 1.0
    return target


def decode(pred, grid, n_boxes=None):
    """pred 已经是 sigmoid 后的 (B,S,S,5B+C)，直接切分解读。
    返回 (B, S, S, n_boxes, 6): [cx, cy, w, h, obj, cls_id]"""
    n_boxes = n_boxes or hp.n_boxes
    B, S = pred.shape[0], pred.shape[1]
    j_idx, i_idx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="xy")
    j_idx, i_idx = j_idx.to(pred.device), i_idx.to(pred.device)

    boxes = []
    for k in range(n_boxes):
        p = pred[..., 5 * k:5 * k + 4]  # (B,S,S,4) x,y,w,h
        obj = pred[..., 5 * k + 4]
        cx = (j_idx[None, :, :, None] + p[..., 0:1]) / S  # 格内偏移 -> 画布坐标
        cy = (i_idx[None, :, :, None] + p[..., 1:2]) / S
        cls_id = pred[..., 5 * n_boxes:].argmax(dim=-1, keepdim=True)
        boxes.append(torch.cat([cx, cy, p[..., 2:4], obj[..., None],
                                cls_id.float()], dim=-1))
    return torch.stack(boxes, dim=3)  # (B,S,S,n_boxes,6)
