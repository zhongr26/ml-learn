"""
迷你 YOLO 网络（YOLOv1 原版风格，为 64x64 输入定制）。

结构契约（后续步骤全部依赖）:
- 输入: (B, 1, canvas, canvas)
- 输出: (B, GRID, GRID, 5*n_boxes + NUM_CLS)
  每格 n_boxes 个框槽位，每槽 5 维 [x,y,w,h,obj]（全 0~1）+ 共享 C 维类别。
  x,y 为格内偏移；w,h 直接回归（v1 风格，无 log/exp 编码）。

关键约束: 总下采样倍数 = canvas / GRID = 16（backbone 4 次 stride=2）。
"""
import torch
from torch import nn

from config import MINI_YOLO_CFG
from src.learn.mini_yolo.synth_data import CANVAS, GRID, NUM_CLS

N_BOXES = MINI_YOLO_CFG.hparams.n_boxes  # 论文 B=2：每格两个框，损失阶段按 IoU 择一负责


class ConvBNAct(nn.Sequential):
    """Conv2d + BatchNorm + SiLU：YOLO 系的标准积木。"""

    def __init__(self, c_in, c_out, k, s):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s,
                      padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),  # swish，比 ReLU 平滑，YOLOv5+ 默认
        )


class MiniYOLONet(nn.Module):
    def __init__(self, num_cls=NUM_CLS, grid=GRID, n_boxes=N_BOXES):
        super().__init__()
        self.grid = grid
        self.num_cls = num_cls
        self.n_boxes = n_boxes
        self.output_dim = grid * grid * (5 * n_boxes + num_cls)

        # backbone：4 个 stride=2 卷积逐级下采样 64 -> 32 -> 16 -> 8 -> 4
        self.backbone = nn.Sequential(
            ConvBNAct(1, 16, 3, 2),  # 64->32
            ConvBNAct(16, 32, 3, 2),  # 32->16
            ConvBNAct(32, 64, 3, 2),  # 16->8
            ConvBNAct(64, 128, 3, 2),  # 8->4 (B, 128, 4, 4)
            ConvBNAct(128, 128, 3, 1),  # 再磨一层特征，不下采样
        )

        # v1 原版：FC 回归整个 S*S*(5B+C)
        # 参考实现是 2048 维特征 -> 4096 -> 1470；这里隐层取 512 够用
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * grid * grid, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feat = self.backbone(x)  # (batch, 128, S, S)
        out = self.classifier(feat)  # (batch, S*S*(5B+C), 4, 4)
        out = out.view(batch_size, self.grid, self.grid, self.num_cls + 5 * self.n_boxes)
        # sigmoid 归一化所有分量到 0~1（参考实现的做法；wh 也被界定在画布内）
        return torch.sigmoid(out)


if __name__ == "__main__":
    net = MiniYOLONet()
    x = torch.randn(2, 1, CANVAS, CANVAS)
    y = net(x)
    assert y.shape == (2, GRID, GRID, 5 * N_BOXES + NUM_CLS), y.shape
    assert (y >= 0).all() and (y <= 1).all()  # sigmoid 全程 0~1
    print(f"输出 {tuple(y.shape)}, 参数量 {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")
