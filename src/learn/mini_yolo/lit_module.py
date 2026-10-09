"""
LitMiniYOLO：Lightning 化的训练壳（对齐 lightning_mnist/model.py 的角色）。

验证除 loss 外还报一个"检测召回@IoU0.5"，训练时能直观看到模型学会找数字。
"""
import lightning as L
import torch

from config import MINI_YOLO_CFG
from src.learn.mini_yolo.loss import MiniYOLOLoss, cxcywh_to_xyxy, pairwise_iou
from src.learn.mini_yolo.mini_net import MiniYOLONet
from src.learn.mini_yolo.targets import encode_labels

hp = MINI_YOLO_CFG.hparams


class LitMiniYOLO(L.LightningModule):
    def __init__(self, lr=None,
                 n_boxes=hp.n_boxes,
                 grid=hp.grid,
                 num_cls=hp.num_cls):
        super().__init__()
        self.save_hyperparameters()
        self.net = MiniYOLONet(n_boxes=n_boxes, num_cls=num_cls, grid=grid)
        self.criterion = MiniYOLOLoss(n_boxes=n_boxes, num_cls=num_cls, grid=grid)

        self.lr = lr or hp.learning_rate
        self.n_boxes = n_boxes
        self.grid = grid
        self.num_cls = num_cls

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, img, labels):
        target = encode_labels(labels, grid=self.grid, num_cls=self.num_cls,
                               n_boxes=self.n_boxes)
        target = target.to(img.device)  # encode 在 CPU 构建，需搬到预测所在设备
        pred = self.forward(img)
        loss = self.criterion(pred, target)
        return pred, target, loss

    def training_step(self, batch, _):
        img, labels = batch
        _, _, loss = self._shared_step(img, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        img, labels = batch
        pred, _, loss = self._shared_step(img, labels)
        self.log('val_loss', loss, prog_bar=True)

        recall = detection_recall(pred, labels, grid=self.grid, n_boxes=self.n_boxes)
        self.log("val_recall50", recall, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@torch.no_grad()
def detection_recall(pred, labels, grid, n_boxes, conf_thr=0.25, iou_thr=0.5):
    """整 batch：被任一高置信预测框(IoU>=0.5)命中的 GT 占比。"""
    from src.learn.mini_yolo.targets import decode
    hits, total = 0, 0
    for b in range(pred.shape[0]):
        boxes = decode(pred[b:b + 1], grid, n_boxes)[0]  # (S,S,n_boxes,6)
        p = boxes.reshape(-1, 6)
        p = p[p[:, 4] > conf_thr]  # obj 过滤
        p = cxcywh_to_xyxy(p[:, :4])
        for gt in labels[b]:
            if gt[4] < 0:
                continue
            total += 1
            g = cxcywh_to_xyxy(gt[:4].unsqueeze(0))
            if len(p) and (pairwise_iou(p, g) >= iou_thr).any():
                hits += 1
    return hits / max(total, 1)
