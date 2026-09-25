# YOLO11n (ultralytics) 面试详解

## 1. 模型概况

| 项 | 说明 |
|---|---|
| 出处 | ultralytics 2024（YOLO 系列第 11 代，YOLOv8 后继） |
| 任务 | 检测 / 分割 / 姿态 / OBB / 分类（本工程用检测） |
| 规模 | **n=2.6M / s=9.4M / m=20M / l=25M / x=56M** |
| 权重 | yolo11n.pt 仅 6.5MB |
| 输入 | 640×640（可变），letterbox 保持长宽比填充 |
| 输出 | 每类边界框 (x1,y1,x2,y2) + 置信度 + 类别 |
| COCO mAP | n≈39.5 / s≈47.0 / m≈51.5（YOLO11n vs YOLOv8n：+1.5 mAP，参数 -8%） |

工业落地最广的检测器：一行命令训练、一行命令导出十几种格式（ONNX/TensorRT/CoreML/TFLite）。

## 2. 架构原理

```
输入 640×640
  Backbone: C3k2 块(SPPF 后) + C2PSA(位置感知注意力)    ← 特征提取
  Neck:     PAFPN (自顶向下 FPN + 自底向上 PAN 融合)     ← 多尺度融合 P3/P4/P5
  Head:     解耦头(分类分支 + 回归分支, anchor-free)      ← 各尺度独立预测
  后处理:   DFL 分布式焦点回归 + NMS / 端到端 NMS-free(可选)
```

关键概念（面试逐个准备）：
- **anchor-free**：不再预设 9 组先验框，直接回归框中心到四边的距离。优点：免聚类 anchor、小数据集不敏感、参数少
- **DFL（Distribution Focal Loss）**：把 bbox 回归建模成离散分布（l×16 bins 概率），期望值即回归结果，边界不确定性更强
- **解耦头**：分类与回归分支分开（耦合头两者冲突：分类要平滑特征、回归要精确空间）
- **C2PSA**：YOLO11 新增，空间+通道双注意力，小目标与密集场景涨点
- **损失**：`box = CIoU + DFL`，`cls = BCE`，两项加权 `3×box + 1×cls`
- **正样本分配 TAL**：Task-Aligned Assigner，按"分类得分×回归 IoU"联合选 top-k 正样本，动态对齐训练目标

## 3. 训练过程（本工程 `train.py`）

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data="coco8.yaml", imgsz=320, epochs=10, device="mps")
```

标准流程：
1. **数据标注**：YOLO 格式 txt（每行 `class x_center y_center w h` 归一化坐标）+ data.yaml（train/val/nc/names）
2. **迁移学习**：加载 COCO 预训练权重，冻结行为由框架自动处理（backbone 前几 epoch 低 lr 预热）
3. **增强**（ultralytics 自动开）：Mosaic 4 图拼接 / CopyPaste / 随机仿射 / HSV / 翻转；**最后 10 epoch 关 Mosaic**（`close_mosaic=10`）防止分布偏移影响收敛
4. **验证**：mAP@0.5 / mAP@0.5:0.95，按置信度-召回曲线自动选 best

**小数据实操**（<500 图）：
- 从 n 而不是 x 开始（大模型小数据必过拟合）
- epochs 100~300、`patience=30` 早停
- 关闭 Mosaic 或减半增强强度：`augment=False` 或逐项配置
- 用 Test-Time Augmentation 评估（推理时翻转/多尺度融合，白捡 1~2 mAP）

## 4. 参数调整

| 超参 | 默认 | 经验 |
|---|---|---|
| lr0 | 0.01 (SGD) | 迁移学习可降 0.001 AdamW；loss 震荡升 warmup |
| imgsz | 640 | 小目标检测升到 1280（显存允许）；速度敏感降到 416 |
| batch | 16 | 大显存开大 batch 稳梯度；M5 建议 16@320 或 8@640 |
| mosaic | 1.0 | 小数据集减到 0.3~0.5 |
| cls/box loss 权重 | 0.5/7.5 | 类别不平衡时升 cls；定位差升 box |
| conf | 0.25 | 误检多上调；漏检多下调（推理参数，不影响训练） |
| iou (NMS) | 0.7 | 密集目标（人群）降到 0.5 防抑制 |

**类别不平衡**：不平衡数据用 `--img-weights`（按类别逆频率采样）或 focal-style 增强；极端不平衡直接复制少数类图片 + 旋转。

## 5. 训练速度优化

- **imgsz 降档**：640→320 计算量约 1/4，早期调参阶段全用小图，最终再大图精调
- **AMP**：ultralytics 默认开 half=True，MPS/CPU 也自动处理
- **cache='ram'**：数据集全量驻内存（M5 24GB 可缓存几千张小图），IO 归零
- **workers=0 on macOS**：spawn 限制，Linux 上开 8
- **多尺度训练 vs 固定尺度**：固定更快；多尺度对部署多分辨率更稳
- **剪枝+微调**：通道剪枝（L1 norm）掉 30~40% 参数，mAP 损失 1~2，再微调 10 epoch 找回

## 6. 推理优化（本工程 `optimize.py`）

| 手段 | 原理 | 实测预期 |
|---|---|---|
| **ONNX Runtime** | 图优化算子融合 | CPU 与 PyTorch 持平或略快；跨平台去 Python 依赖 |
| **ONNX int8 动态量化** | 权重 QInt8 | 模型 6.5MB→1.7MB，CPU 提速 1.5~2.5x，mAP 掉 0.5~1.5 |
| **TensorRT** (NVIDIA) | int8 + 层融合 + kernel auto-tune | GPU 上 3~5x（Mac 无，但面试必背） |
| **CoreML** | macOS 原生 ANE/GPU | M 芯片部署首选，本工程演示导出 |
| **ncnn/TFLite** | 移动端 | 手机端 8ms@FP16 |
| **batch + 视频跳帧** | 多路视频拼 batch；抽帧检测 | 吞吐线性提升 |
| **输入下采样 + 模型换小** | n@416 比 x@640 快 10x+ | 速度优先时先减 imgsz 再换大模型 |

**导出一张表（面试背）**：`model.export(format="onnx/coreml/engine/tflite/ncnn/openvino")`，全部半精度选项 + int8 选项，导出后用对应 runtime 加载。

## 7. 面试 Q&A

**Q1: anchor-free 为什么成为主流？**
A: ① anchor 超参依赖聚类、换数据集要重调 ② 正负样本极度不平衡 ③ 框大小预测受先验限制。anchor-free 直接回归点到边距离，结合 TAL 动态分配，精度不掉且工程简化。

**Q2: NMS 原理？端到端 NMS-free 怎么做？**
A: NMS 按置信度排序，抑制与最高分框 IoU>阈值的其他框。密集场景误杀（重叠目标），可用 Soft-NMS（线性衰减分数）。新趋势：YOLOv10/RT-DETR 用一对一分配 + 一致双分配，推理免 NMS，延迟更稳（无后处理分支）。

**Q3: mAP@0.5 和 mAP@0.5:0.95 区别？**
A: @0.5 单一 IoU 阈值偏松；0.5:0.95 是 10 个 IoU 阈值均值，衡量定位精度梯度。报告务必给后者，只报 @0.5 有粉饰嫌疑。

**Q4: 小目标检测效果差怎么救？**
A: ① imgsz 提到 1280+ ② 用 P2 检测头（加高分辨率分支）③ 数据增强保留小目标切片(slicing)、用 SAHI 切图推理 ④ CIoU 对小目标梯度不稳，可用 NWD 等度量辅助。

**Q5: 训练 loss 下降但 mAP 不升？**
A: 常见于：过拟合（val loss 反升）、置信度校准漂移（检查 conf 曲线）、增强太强（最后关 Mosaic）、正样本分配震荡（降 lr）。用 val 曲线 + PR 曲线定位，别只盯 train loss。

**Q6: 视频流实时检测怎么设计？**
A: 检测每 N 帧跑一次 + 中间帧用跟踪（ByteTrack）传播框；跟踪 ID 关联用 IoU + ReID 特征；检测与跟踪异步流水线，检测帧率可以低于播放帧率。
