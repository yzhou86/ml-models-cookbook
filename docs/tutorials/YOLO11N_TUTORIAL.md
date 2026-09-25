# YOLO11n 目标检测实战教程：训练、参数、推理与调优

本教程介绍 YOLO11n 的数据格式、迁移训练、目标检测推理、ONNX 静态量化和 CoreML 导出，默认环境为 MacBook Air M5、24GB 统一内存。

对应代码：

- [`yolo11n/train.py`](../../yolo11n/train.py)：Ultralytics 训练入口
- [`yolo11n/infer.py`](../../yolo11n/infer.py)：检测和结果可视化
- [`yolo11n/optimize.py`](../../yolo11n/optimize.py)：PyTorch、ONNX、int8 和 CoreML
- [`yolo11n/README.md`](../../yolo11n/README.md)：架构与面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 解释 Backbone、Neck、检测头、DFL 和 NMS 的职责。
2. 准备 YOLO 格式数据集和 YAML 配置。
3. 在 MPS 上完成 YOLO11n 微调。
4. 理解 imgsz、batch、学习率、冻结层、置信度和早停。
5. 使用训练后的 `best.pt` 推理图片、视频或目录。
6. 公平比较 PyTorch MPS、PyTorch CPU、ONNX CPU 和 ONNX int8。

## 2. YOLO 检测链路

```text
图片
  → letterbox resize/padding
  → Backbone 提取多层视觉特征
  → Neck 融合不同尺度
  → Head 输出类别和框分布
  → 框解码
  → 置信度过滤
  → NMS
  → 最终 boxes/classes/scores
```

YOLO11n 中的 `n` 表示 nano，是该系列最小的版本，约 2.6M 参数，适合端侧实验和小数据迁移。

### 2.1 多尺度特征

较浅的高分辨率特征有利于小目标，较深的低分辨率特征具有更强语义。Neck 把不同尺度信息融合，让检测头同时处理大小不同的目标。

### 2.2 Anchor-free 和 DFL

当前 YOLO 检测头采用 anchor-free 思路，预测候选位置到边界框四条边的距离，而不是依赖预先聚类的 anchor 尺寸。

DFL 把每条边的距离看成离散分布，再根据期望得到连续位置。它可以表达边界的不确定性，通常比直接回归单个数值更细致。

### 2.3 NMS

模型可能对同一物体输出多个重叠框。NMS 按置信度排序，保留最高分框，并抑制与它 IoU 过高的候选框。

密集目标场景中，NMS 可能误删真实的相邻目标，因此置信度和 IoU 阈值需要在目标数据上调节。

## 3. 第一次推理

```bash
cd /Users/yu/Code/ml-models-cookbook
uv run python -m yolo11n.infer
```

脚本会下载 `yolo11n.pt` 和 bus 示例图片，自动选择 MPS，并把结果保存到：

```text
samples/yolo_result.jpg
```

测试自己的图片：

```bash
uv run python -m yolo11n.infer \
  --src /绝对路径/test.jpg \
  --weights yolo11n.pt \
  --conf 0.35 \
  --imgsz 640 \
  --device mps
```

`--src` 也可以是视频或目录，Ultralytics 会自动处理对应输入。

## 4. 准备自定义数据

常见目录结构：

```text
data/my_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每张图片对应同名 `.txt` 标签，例如 `images/train/001.jpg` 对应 `labels/train/001.txt`。

每行标签格式：

```text
class_id x_center y_center width height
```

四个坐标都除以图像宽高，归一化到 `[0, 1]`。

示例：

```text
0 0.512 0.438 0.220 0.350
1 0.231 0.620 0.105 0.180
```

数据 YAML：

```yaml
path: /absolute/path/data/my_dataset
train: images/train
val: images/val

names:
  0: person
  1: helmet
```

检查重点：

- class id 从 0 开始且连续。
- 坐标没有超出 `[0, 1]`。
- 图片和标签文件一一对应。
- 空目标图片可以没有标签内容，但必须符合数据加载规则。
- 同一视频的相邻帧不能随机分到训练集和验证集，否则会产生数据泄漏。

## 5. 第一次训练：coco8 冒烟实验

`coco8.yaml` 是 Ultralytics 的 8 张图片玩具数据，只用于跑通流程：

```bash
uv run python -m yolo11n.train \
  --data coco8.yaml \
  --weights yolo11n.pt \
  --epochs 1 \
  --imgsz 64 \
  --batch 2 \
  --optimizer AdamW \
  --lr0 1e-3 \
  --workers 0 \
  --device mps
```

1 个 epoch 和 8 张图的 mAP 没有业务意义。它只验证数据下载、模型构建、MPS 训练、验证和权重保存。

## 6. 第一次真实训练

```bash
uv run python -m yolo11n.train \
  --data data/my_dataset.yaml \
  --weights yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --optimizer AdamW \
  --lr0 1e-3 \
  --weight-decay 5e-4 \
  --momentum 0.937 \
  --patience 20 \
  --workers 0 \
  --device mps \
  --freeze 0
```

Ultralytics 默认把结果保存在类似目录：

```text
runs/detect/train/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── results.png
├── confusion_matrix.png
└── ...
```

- `best.pt`：验证指标最好的 checkpoint，通常用于部署。
- `last.pt`：最后一个 epoch，用于续训或分析。

## 7. 训练参数怎么调

### 7.1 `--imgsz`

图像尺寸是速度和小目标精度的关键参数。卷积计算量近似随面积变化：

\[
compute\propto imgsz^2
\]

从 640 降到 320，主体卷积计算量大约降到四分之一。

建议：

- 64/320：冒烟和早期调参。
- 640：常规训练和部署基线。
- 960/1280：小目标场景，但训练和推理明显变慢。

提高分辨率前，应先确认原图中目标确实太小，而不是标注或类别定义有问题。

### 7.2 `--batch`

M5 24GB 可从以下配置起步：

- `imgsz=320, batch=16`
- `imgsz=640, batch=8`
- 内存压力大时降低到 4 或 2

统一内存同时被系统和 MPS 使用。出现明显 swap 时，应降低 batch。

### 7.3 `--optimizer` 和 `--lr0`

本工程默认 AdamW 和 `lr0=1e-3`：

- 小数据迁移：AdamW `1e-4`～`1e-3`。
- 数据充足、较大 batch：可比较 SGD。
- loss 剧烈震荡：降低 `lr0`。
- loss 不下降：先检查标签，再调整学习率。

不能脱离 batch、warmup 和预训练状态孤立比较学习率。

### 7.4 `--weight-decay`

默认 `5e-4`，用于抑制过拟合。值过大可能欠拟合，过小则可能让小数据训练集记忆过强。

### 7.5 `--momentum`

默认 `0.937`。它对 SGD 的速度平滑很重要；AdamW 也有内部动量参数，Ultralytics 会按优化器配置处理。

### 7.6 `--patience`

验证指标连续若干 epoch 没有改善时提前停止：

- 小数据波动大：适当提高到 20～50。
- 快速实验：设为 5～10。
- 早停依据验证指标，不是训练 loss。

### 7.7 `--freeze`

冻结模型前 N 层，例如：

```bash
--freeze 10
```

冻结可以降低反向计算和过拟合风险，但 N 对应的是模型内部层顺序，具有架构依赖性。应检查实际被冻结的层，而不是把数字当作通用语义。

小数据可以先冻结部分骨干，再用较低学习率全量微调。

### 7.8 `--workers`

macOS 先用 0，确认稳定后测试 2 或 4。如果 MPS 等待数据、CPU 预处理成为瓶颈，增加 workers 可能提高吞吐；小数据下多进程启动成本可能更高。

## 8. 如何读训练结果

检测模型不能只看 loss。重点指标包括：

### 8.1 Precision 和 Recall

- Precision 高：预测出来的目标大多正确。
- Recall 高：真实目标大多被找到。

提高推理置信度通常提升 precision、降低 recall。

### 8.2 mAP@0.5

只在 IoU=0.5 的条件下计算 AP，定位要求相对宽松。

### 8.3 mAP@0.5:0.95

在 0.5、0.55、...、0.95 十个 IoU 阈值上取平均，更严格地衡量定位质量。正式报告至少应包含这一项。

### 8.4 每类指标

总体 mAP 可能掩盖少数类问题。还应查看：

- 每类 AP。
- 混淆矩阵。
- 按目标尺寸统计的效果。
- 不同场景、光照和相机的切片指标。

## 9. 使用训练后的模型推理

```bash
uv run python -m yolo11n.infer \
  --weights runs/detect/train/weights/best.pt \
  --src /绝对路径/test.jpg \
  --conf 0.35 \
  --imgsz 640 \
  --device mps
```

### 9.1 `--conf`

- 误检多：提高 conf。
- 漏检多：降低 conf。
- 不同类别可能需要不同阈值。

最佳阈值应根据验证集 PR 曲线和业务成本选择，而不是固定使用 0.35。

### 9.2 推理尺寸必须和部署一致

可以用 640 训练、320 推理，但小目标召回可能下降。正式评估必须使用实际部署的 `imgsz`。

## 10. 数据增强怎么理解

Ultralytics 会管理 Mosaic、颜色、翻转和几何增强。增强不是越强越好：

- 小数据需要增强减少过拟合。
- 过强几何变换可能制造不真实目标。
- 文字、方向性物体不一定适合水平翻转。
- 训练后期通常应减弱强增强，使数据更接近部署分布。

当前封装脚本没有暴露全部增强参数。需要精调时，可扩展 CLI 或直接使用 Ultralytics 配置文件，但必须记录最终配置。

## 11. 性能优化实验

先运行 PyTorch、ONNX 和 int8：

```bash
uv run python -m yolo11n.optimize \
  --weights yolo11n.pt \
  --imgsz 640 \
  --runs 20 \
  --calib-dir data/detection_calibration \
  --calib-samples 100 \
  --rebuild-int8
```

脚本比较：

1. PyTorch CPU 端到端延迟。
2. PyTorch MPS 端到端延迟。
3. ONNX Runtime CPU 端到端延迟。
4. ONNX QDQ int8 CPU 端到端延迟。

所有路径通过 Ultralytics 的预测接口，使用相同图片、尺寸、解码和 NMS，避免只比较裸网络而忽略前后处理。

### 11.1 ONNX 导出

脚本导出固定输入尺寸的 ONNX，并进行图简化。ONNX 的价值包括：

- 跨框架部署。
- ONNX Runtime 图优化。
- 更清晰的 CPU 线程和 provider 管理。
- 作为静态 PTQ 的输入。

### 11.2 静态 QDQ PTQ

目标检测以卷积为主，因此使用静态量化：

```text
FP32 ONNX
  → 代表性图片校准激活范围
  → 插入 QuantizeLinear / DequantizeLinear
  → ONNX int8 模型
```

当前实现保留检测头为 FP32，只量化更稳定的 Backbone/Neck。检测头包含 DFL、sigmoid 和坐标解码，对量化范围敏感；盲目量化曾会导致无效置信度或严重框偏移。

这是混合精度的典型思路：不追求“所有节点都是 int8”，而追求正确输出、速度和精度的最佳平衡。

### 11.3 校准集

推荐 100～500 张来自真实部署分布的图片，覆盖：

- 不同目标类别和尺寸。
- 白天、夜间和逆光。
- 不同相机、压缩和背景。
- 空目标和密集目标场景。

校准集不需要标签，但量化后的 mAP 评估集必须有标签。

如果不传 `--calib-dir`，脚本只用 bus 图片完成 smoke test，不能据此评价 mAP。

### 11.4 `--rebuild-int8`

量化模型存在时脚本会复用它。更换校准集、输入尺寸、ONNX 图或量化策略后必须加：

```bash
--rebuild-int8
```

否则可能误用旧的 int8 文件。

## 12. CoreML 导出

先安装可选依赖：

```bash
uv sync --python 3.12 --extra apple
```

再导出：

```bash
uv run python -m yolo11n.optimize \
  --weights yolo11n.pt \
  --imgsz 640 \
  --runs 20 \
  --coreml
```

脚本请求 FP16 CoreML 并包含 NMS。导出成功不等于部署完成，还应在实际 Swift/CoreML 环境验证：

- 输入颜色和归一化。
- 坐标映射与 letterbox 还原。
- NMS 输出语义。
- 数值精度。
- 首次加载和稳定延迟。

## 13. 进一步优化方向

### 13.1 先调整模型和分辨率

通常最有效的杠杆是：

1. 选择 n/s/m 模型规模。
2. 调整 imgsz。
3. 再选择运行时和精度。

直接换量化引擎之前，先检查更小输入是否已经满足精度。

### 13.2 视频检测加跟踪

不必每帧都运行检测：

```text
每 N 帧运行检测
  → ByteTrack 等跟踪器传播框
  → 场景变化或跟踪失效时重新检测
```

这通常比单纯优化每一帧的检测模型更有效。

### 13.3 切图检测

超高分辨率的小目标场景可使用 overlapping tiles 或 SAHI。它提升小目标像素占比，但增加推理次数和跨 tile 去重成本。

## 14. 常见故障

### mAP 很低

- 标签坐标格式错误。
- class id 与 YAML names 不一致。
- 训练/验证分布不一致。
- 图像中目标太小。
- 数据量太少或标签漏标严重。

### loss 下降但 mAP 不升

- 模型正在过拟合训练集。
- 验证集太小或有数据泄漏。
- 增强过强，训练分布不真实。
- 类别分数提升但框定位没有改善。

### MPS 比 CPU 慢

模型很小、输入尺寸低或只测单张图时，MPS 调度开销可能占主导。必须预热和同步后比较端到端延迟。

### int8 输出异常

- 确认使用最新 `--rebuild-int8`。
- 增加代表性校准图片。
- 检查是否误量化检测头敏感节点。
- 检查输出置信度是否仍在合理范围，并重新计算 mAP。

## 15. 面试表达模板

> 我使用 YOLO11n 在 M5 上完成目标检测迁移训练。数据采用 YOLO 归一化框格式，并按视频或采集会话划分训练和验证集，避免相邻帧泄漏。训练阶段先用 320 分辨率快速调参，再用部署分辨率训练和验证，重点观察 mAP@0.5:0.95、每类 AP 和 PR 曲线。部署时对 PyTorch CPU、MPS、ONNX Runtime 和 ONNX int8 做相同预处理、解码及 NMS 的端到端比较。静态 PTQ 使用真实图片校准，并把 DFL、sigmoid 和坐标解码所在的检测头保留为 FP32，避免量化导致框和置信度失真。

常见追问：

- 为什么用 mAP@0.5:0.95：它比单一 IoU=0.5 更严格地评价定位质量。
- 为什么小目标提高 imgsz 有用：目标在特征图上占据更多像素和网格。
- 为什么量化要保留检测头：框分布和坐标解码对数值误差敏感。
- 为什么 benchmark 要包含 NMS：用户感受到的是完整预测延迟，而不是裸网络时间。
- 为什么不能随机拆视频帧：相邻帧高度相似，会造成验证集泄漏。

## 16. 建议练习顺序

1. 使用 COCO 权重完成 bus 图片推理。
2. 用 coco8 完成 1 epoch 冒烟训练。
3. 准备一个两类别自定义数据集并检查标签。
4. 用 320 分辨率搜索 batch 和学习率。
5. 用 640 分辨率训练最终候选模型。
6. 使用 `best.pt` 推理独立测试图片。
7. 比较 PyTorch CPU、MPS、ONNX 和 int8。
8. 使用真实校准集重新构建 int8，并验证 mAP。
9. 可选导出 CoreML，在原生环境验证端到端结果。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
