# yolo-learn 学习日志

## 当前状态（2026-10-07）
- 阶段 5（基础整合）完成：入口 `python -m src.learn.yolo_learn.__main__` 串起 训练→数据信息→推理可视化→TorchScript 导出。
- 工程修正：
  - 权重统一放 `checkpoints/yolo/`（`resolve_weights()` 用 ultralytics 的 `attempt_download_asset` 直接下载到目标路径，不再散落 cwd/src）。
  - `YOLOModel.predict` 不再存盘，带框结果图由 viz 统一保存到 `outputs/yolo/inference/`，消除重复文件。
  - `__main__` 中 `YOLOModel(best)` 使用训练产出的 best 权重（此前误用预训练权重）。
  - AMP 默认关闭（`YOLO_AMP`，MX450 上混合精度出 NaN）；`YOLO_BATCH_SIZE<=0` 时 autobatch（-1）。
- 阶段 1 完成：`src/learn/yolo_learn/` 模块搭好，对齐 lightning_mnist 架构（data/model/train/io_utils/viz/__main__）。
- COCO 数据：默认 `coco8.yaml`（迷你验证集，首次训练自动下载到 `datasets/yolo/`）；`.env.yolo` 中 `YOLO_DATA_YAML=coco.yaml` 切完整版（约 19GB）。
- 已验证：模块导入、YOLO 推理、config 解析（OS env > .env.yolo > .env > 默认值）。训练全流程未跑（按要求跳过），首次运行 `python -m src.learn.yolo_learn.__main__` 即可。
- 环境：`ultralytics==8.4.174`；网络访问 ultralytics.com 不通（示例图片下载超时），GitHub 权重下载可用。
- 依赖坑：ultralytics 会拉 opencv>=5/4.14 和 numpy 2.x，与本项目 numpy<2.0 冲突，requirements.txt 已固定 `opencv-python==4.10.0.84`。
- 已知问题：coco8 + 大 batch 训练时出现 loss NaN（数据太少），真实训练建议完整 COCO 或合理 batch。

## 计划
1. ✅ ultralytics 推理示例 + COCO 数据接入
2. 自定义数据集训练
3. 手写 IoU/NMS/CIoU/mAP
4. 迷你 YOLO 从零实现
5. 可视化与整合

## 模块结构（对齐 lightning_mnist）
- `data.py` — prepare_data()/dataset_info()/sample_images()：把 ultralytics 的 datasets_dir 指向 `datasets/yolo/`，`check_det_dataset` 触发下载
- `model.py` — `YoloModel`：薄封装（predict 统一读 config 超参），`boxes_to_list` 把 Results 转结构化列表
- `train.py` — 与 mnist 相同的重训策略（best.pt 存在且未 RETRAIN 则跳过）；产物拷贝到 `checkpoints/yolo/best.pt`
- `io_utils.py` — TorchScript 导出重载 demo（ultralytics 导出后仍用 `YOLO(path)` 加载，接口一致）
- `viz.py` — 推理结果图 + 类别分布条形图，存 `outputs/yolo/`
- 注意：config 的单例 `YOLO` 在同时 `from ultralytics import YOLO` 的文件里要 `from config import YOLO as YOLO_CFG`

## 学习要点
- 目标检测基础：bbox `(x, y, w, h)`、IoU = 交并比、NMS 去重、mAP@0.5 精度指标、anchor 预设框、one-stage（YOLO 系）直接回归检测 vs two-stage（Faster R-CNN）先提候选区。
- ultralytics Results：`r.boxes.xyxy / xywh(n归一化) / conf / cls`，`r.boxes.data` 一步到位 (N,6)=[x1,y1,x2,y2,conf,cls]；`r.plot()` 返回带框 BGR 图。
- 型号后缀 n/s/m/l/x = 模型规模递增，推理先用 n。
- 训练产物：`project/name/weights/{best,last}.pt` + results.png；`model.trainer.best` 取 best 路径。
- ultralytics 全局设置（`from ultralytics import settings`）控制 datasets_dir 等下载位置，写入用户 AppData。
