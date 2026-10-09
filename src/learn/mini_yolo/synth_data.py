"""
合成检测数据集：把 MNIST 数字随机贴到画布上，构造目标检测任务。

每张样本:
- 画布 canvas x canvas 灰度图，随机贴 1~max_objs 个 MNIST 数字
  （28x28 -> 缩放到 digit_size）
- 标签为数字的包围框 (cx, cy, w, h，均相对画布归一化到 0~1) + 类别(0~9)

设计要点:
- 检测模型需要"哪里有东西 + 是什么"，所以标签必须带几何信息，
  这与分类数据集（只有类别）的本质区别
- 框用 (中心点+宽高) 归一化存储 —— YOLO 系的标准标签格式，
  与画布尺寸无关，换分辨率不用改标注
- 尺寸/规模/损失权重全部来自 config（MINI_YOLO_ 前缀环境变量可覆盖）
"""
import lightning as L
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from config import MINI_YOLO_CFG, MNIST_CFG

hp = MINI_YOLO_CFG.hparams
CANVAS = hp.canvas      # 画布边长（模块级别名，供网络/编码解码引用）
DIGIT = hp.digit_size   # 数字缩放后的边长
GRID = hp.grid          # 网格数 S
NUM_CLS = hp.num_cls


class SynthDetDataset(Dataset):
    """MNIST -> 合成检测样本。输出 (1,canvas,canvas) 图像 + (max_objs,5) 标签。"""

    def __init__(self,
                 root=MNIST_CFG.data_dir,  # 复用已下载的 MNIST 数据
                 train=True,
                 n_per_epoch=None,
                 max_objs=None):
        super().__init__()
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=T.Compose([
                T.Resize((DIGIT, DIGIT)),
                T.ToTensor(),  # (1, digit, digit) 0~1
            ])
        )
        self.n_per_epoch = n_per_epoch or (hp.n_train if train else hp.n_val)
        self.max_objs = max_objs or hp.max_objs

    def __len__(self):
        return self.n_per_epoch

    def __getitem__(self, idx):
        roll = torch.rand(1).item()
        canvas = torch.zeros(1, CANVAS, CANVAS)
        labels = []
        # 随机决定贴几个数字（1 或 max_objs 个，控制重叠复杂度）
        n_objs = 1 if roll < 0.5 else self.max_objs
        for _ in range(n_objs):
            k = torch.randint(0, len(self.mnist), (1,)).item()
            img, cls = self.mnist[k]  # (1,32,32), 标量

            # 数字左上角随机放置（留出边距，避免框出界）
            x0 = torch.randint(0, CANVAS - DIGIT + 1, (1,)).item()
            y0 = torch.randint(0, CANVAS - DIGIT + 1, (1,)).item()
            canvas[:, y0:y0 + DIGIT, x0:x0 + DIGIT] = \
                torch.max(canvas[:, y0:y0 + DIGIT, x0:x0 + DIGIT], img)  # 重叠区取亮

            # (左上角像素坐标) -> (归一化中心点+宽高)
            cx = (x0 + DIGIT / 2) / CANVAS
            cy = (y0 + DIGIT / 2) / CANVAS
            labels.append([cx, cy, DIGIT / CANVAS, DIGIT / CANVAS, float(cls)])

        # 定长标签：不足 max_objs 用 -1 填充（collate 需要固定形状）
        labels = torch.tensor(labels, dtype=torch.float32)
        pad = torch.full((self.max_objs - len(labels), 5), -1.0)
        return canvas, torch.cat([labels, pad])


class SynthDetDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size=hp.batch_size,
                 num_workers=hp.num_workers):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = SynthDetDataset(train=True)
            self.val_dataset = SynthDetDataset(train=False)
        if stage in ("test", None):
            self.test_dataset = SynthDetDataset(train=False)

    def _loader(self, ds, shuffle):
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(ds, **kwargs)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
