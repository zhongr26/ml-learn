"""
迷你 YOLO 损失 —— YOLOv1 论文公式的忠实实现（对照 pytorch-YOLO-v1-master/yoloLoss.py）。

loss = λ_coord * [xy 的 MSE + sqrt(wh) 的 MSE]     只算负责框
     + λ_obj  * (obj - IoU)^2                      只算负责框（target 是实际 IoU，不是 1！）
     + 1.0    * (obj - 0)^2                        有物体格子里的"不负责框"
     + λ_noobj * (obj - 0)^2                       无物体格子
     + (cls_prob - one_hot)^2                      有物体格子的类别

前置约定: pred/target 都是 sigmoid 后的 (B, S, S, 5*B_box + C)，
与 mini_net.forward 输出、targets.encode_labels 的 target 同形状。
"""
import torch
from torch import nn

from config import MINI_YOLO_CFG

hp = MINI_YOLO_CFG.hparams


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) [cx,cy,w,h] -> [x1,y1,x2,y2]，坐标均为画布归一化值。"""
    return torch.cat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2
    ], dim=-1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N,4) vs (M,4) xyxy -> (N,M) IoU。与 core.py 的 box_iou 相同"""
    lt = torch.max(a[..., None, :2], b[..., None, :2])
    rb = torch.min(a[..., None, 2:], b[..., None, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])[:, None] \
           + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])[:, None]
    return inter / (area - inter + 1e-9)


class MiniYOLOLoss(nn.Module):
    def __init__(self, grid=None, n_boxes=None, num_cls=None,
                 lambda_coord=None, lambda_noobj=None, lambda_obj=2.0):
        super(MiniYOLOLoss, self).__init__()
        self.S = grid or hp.grid
        self.B = n_boxes or hp.n_boxes
        self.C = num_cls or hp.num_cls
        self.l_coord = lambda_coord or hp.lambda_box
        self.l_noobj = lambda_noobj or hp.lambda_noobj
        self.l_obj = lambda_obj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = pred.shape[0]
        S, B, C = self.S, self.B, self.C

        # 1) 按照格子分区
        has_obj = target[..., 4] > 0  # (b, S, S) 框1槽位有 obj

        # 2) B 框择优
        # 先展平 M=有格子数
        bh, ih, jh = has_obj.nonzero(as_tuple=True)  # 三个 (M,) 消去 b,S,S
        gt_cxcywh = target[bh, ih, jh, 0:4]  # (M, 4)
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)  # (M, 4)

        # pred 每格布局为 B 个 [x,y,w,h,obj] 槽位 + C 类，先按槽位拆开
        pred_slots = pred[..., :5 * B].view(*pred.shape[:3], B, 5)  # (b,S,S,B,5)
        pred_cxcywh = pred_slots[bh, ih, jh, :, :4]  # (M, B, 4)
        pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)  # (M, B, 4)

        ious = torch.stack(
            [pairwise_iou(gt_xyxy, pred_xyxy[:, k]).diagonal() for k in range(B)],
            dim=1
        )  # (M, B) --> 第 k 个框与 GT 的 IoU
        best_iou, best_k = ious.max(dim=1)

        # 3) 负责(resp)的框的坐标损失
        resp_pred_cxcywh = pred_cxcywh[torch.arange(len(bh)), best_k]  # (M, 4)
        loc_loss = \
            ((resp_pred_cxcywh[:, :2] - gt_cxcywh[:, :2]) ** 2).sum() \
            + ((resp_pred_cxcywh[:, 2:4].sqrt() - gt_cxcywh[:, 2:4].sqrt()) ** 2).sum()

        # 4) 负责(resp)的框的置信度损失
        resp_obj = pred[bh, ih, jh, 5 * best_k + 4]  # (M, ) 胜出的框的 obj
        contain_loss = ((resp_obj - best_iou) ** 2).sum()

        # 5) 不负责的背景框的置信度损失
        # 有物体格子里另外 B-1 个框；用掩码一次算齐所有格的所有框
        is_resp = (torch.arange(B, device=pred.device)[None, :]
                   == best_k[:, None])  # (M, B)
        all_obj = pred.new_zeros(batch, S, S, B)
        for k in range(B):
            all_obj[..., k] = pred[..., 5 * k + 4]  # (b,S,S,B) 每框的 obj 通道
        cell_resp = torch.zeros_like(all_obj)
        cell_resp[bh, ih, jh] = is_resp.float()
        cell_has = torch.zeros_like(all_obj)
        cell_has[bh, ih, jh] = 1.0

        # 不负责且在有物体的格子： (pred_obj-0)^2
        not_contain_loss = ((all_obj * cell_has * (1 - cell_resp)) ** 2).sum()
        # 无物体的格子： (pred_obj-0)^2 * λ_noobj
        noobj_loss = ((all_obj * (1 - cell_has)) ** 2).sum() * self.l_noobj

        # 6) 类别损失
        cls_pred = pred[bh, ih, jh, B * 5:]  # (M, C)
        cls_target = target[bh, ih, jh, B * 5:]
        cls_loss = ((cls_pred - cls_target) ** 2).sum()

        total = (self.l_coord * loc_loss
                 + self.l_obj * contain_loss
                 + not_contain_loss + noobj_loss + cls_loss)
        return total / batch
