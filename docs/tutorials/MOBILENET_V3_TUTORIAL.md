# MobileNetV3 图像分类实战教程：训练、参数、推理与调优

本教程介绍 MobileNetV3 的迁移学习、图像分类推理和量化优化，默认环境为 MacBook Air M5、24GB 统一内存。

对应代码：

- [`mobilenet_v3/train.py`](../../mobilenet_v3/train.py)：ImageFolder 迁移学习
- [`mobilenet_v3/infer.py`](../../mobilenet_v3/infer.py)：ImageNet 预训练模型推理
- [`mobilenet_v3/optimize.py`](../../mobilenet_v3/optimize.py)：动态量化、静态 PTQ、MPS 和 compile
- [`mobilenet_v3/README.md`](../../mobilenet_v3/README.md)：架构与面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 解释 depthwise separable convolution、inverted residual、SE 和 h-swish。
2. 使用 ImageFolder 数据完成迁移学习。
3. 正确冻结骨干和 BatchNorm。
4. 调整分辨率、batch、学习率、weight decay 和 AMP。
5. 使用预训练模型完成 Top-5 推理。
6. 解释动态量化为什么不适合卷积主干，以及静态 PTQ 为什么需要校准集。

## 2. MobileNetV3 为什么适合端侧

MobileNetV3 结合了轻量卷积结构、神经架构搜索和硬件友好激活。典型模块为：

```text
输入
  → 1×1 expansion
  → depthwise convolution
  → squeeze-and-excitation（部分 block）
  → 1×1 projection
  → residual（形状一致时）
```

### 2.1 Depthwise separable convolution

普通卷积计算量大致为：

\[
H\times W\times K^2\times C_{in}\times C_{out}
\]

Depthwise + Pointwise 的计算量约为：

\[
H\times W\times K^2\times C_{in}
+H\times W\times C_{in}\times C_{out}
\]

它把空间卷积和通道混合拆开，从而显著减少计算量。

### 2.2 Inverted residual

传统残差网络常先压缩再扩展；MobileNetV2/V3 先把通道扩展，在高维空间做 depthwise convolution，再投影回较低维。残差连接位于窄通道之间，因此称为 inverted residual。

### 2.3 SE 和 h-swish

- SE 根据全局通道统计学习通道权重，让模型强调重要通道。
- h-swish 是 swish 的分段近似，比 sigmoid 组合更适合部分移动端实现。

## 3. small 和 large 怎么选

| 版本 | 规模 | 使用建议 |
|---|---|---|
| small | 约 2.5M 参数 | 极低延迟、快速实验、小数据 |
| large | 约 5.4M 参数 | 精度更高，仍属于轻量模型 |

训练脚本默认 `small`，优化脚本固定演示 `large`。优化脚本使用 torchvision ImageNet 权重，不会自动读取训练脚本生成的自定义分类 checkpoint。

## 4. 第一次推理

```bash
cd /Users/yu/Code/ml-models-cookbook
uv run python -m mobilenet_v3.infer
```

脚本会下载示例犬类图片，使用 MobileNetV3-Small 的 ImageNet 权重，并输出 Top-5 类别。

测试自己的图片：

```bash
uv run python -m mobilenet_v3.infer \
  --variant large \
  --src /绝对路径/image.jpg
```

推理预处理直接使用 `weights.transforms()`，它封装了对应权重要求的 resize、crop、tensor 转换和 ImageNet normalization。预处理必须和权重匹配，否则即使模型正常运行，准确率也会显著下降。

## 5. 准备训练数据

训练脚本使用 torchvision `ImageFolder`：

```text
data/imagenet_like/
├── cat/
│   ├── 001.jpg
│   └── 002.jpg
├── dog/
│   ├── 001.jpg
│   └── 002.jpg
└── bird/
    └── ...
```

目录名就是类别名。真实项目建议分开保存：

```text
data/classification/
├── train/class_a/...
├── train/class_b/...
├── val/class_a/...
├── val/class_b/...
└── test/...
```

当前教学脚本只读取一个目录并训练，没有内置验证循环。因此正式项目应增加独立验证脚本或扩展训练代码。不能用训练准确率代替泛化能力。

## 6. 第一次训练：合成数据冒烟实验

当 `--data` 目录不存在时，脚本生成 128 张随机图像和 4 个随机类别，用来验证训练链路：

```bash
uv run python -m mobilenet_v3.train \
  --data data/not-exist \
  --variant small \
  --freeze \
  --imgsz 64 \
  --batch 16 \
  --epochs 1 \
  --out runs/mobilenet-smoke.pt
```

随机数据没有可学习规律，准确率没有业务意义。这一步只检查 MPS、前后向、AMP、优化器和 checkpoint 保存。

## 7. 第一次真实迁移学习

先冻结特征骨干，只训练新分类头：

```bash
uv run python -m mobilenet_v3.train \
  --data data/classification/train \
  --variant small \
  --freeze \
  --imgsz 160 \
  --batch 32 \
  --epochs 10 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --workers 0 \
  --device mps \
  --amp \
  --seed 42 \
  --out runs/mobilenet-small-head.pt
```

然后用更低学习率解冻全模型：

```bash
uv run python -m mobilenet_v3.train \
  --data data/classification/train \
  --variant small \
  --imgsz 160 \
  --batch 16 \
  --epochs 10 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --workers 0 \
  --device mps \
  --amp \
  --seed 42 \
  --out runs/mobilenet-small-full.pt
```

当前脚本的两次运行都会从 torchvision 预训练权重开始，不会自动把第一阶段 checkpoint 接到第二阶段。若要执行严格的两阶段连续训练，需要加载第一阶段 `state_dict` 后再解冻训练。现有两条命令适合作为“冻结与全量微调”的独立对照实验。

## 8. 训练数据增强

当前训练预处理为：

```text
RandomResizedCrop(imgsz, scale=(0.7, 1.0))
RandomHorizontalFlip()
ToTensor()
ImageNet Normalize
```

`RandomResizedCrop` 同时改变裁剪区域和尺度，可以提升尺度鲁棒性。水平翻转适合普通物体分类，但不适合具有左右语义的任务，例如文字、交通方向或特定医疗影像。

训练和验证增强必须分开：验证集通常只使用 deterministic resize/center crop 和相同 normalization。

## 9. 训练参数怎么调

### 9.1 `--imgsz`

输入分辨率会近似以平方关系影响大部分卷积计算：

\[
compute\propto H\times W
\]

建议：

- 128/160：快速调参、目标较大。
- 192/224：更重视细节和精度。
- 小目标或细粒度分类：提高分辨率可能有效，但应实测。

先用 160 找到可行超参数，再用 224 做最终训练通常更节省时间。

### 9.2 `--batch`

M5 24GB 上 small@160 可从 32 开始，large@224 可从 16 开始。实际可用 batch 取决于系统同时运行的软件和是否全量微调。

更大 batch 不一定更准确；它主要改变吞吐和梯度噪声，并可能需要调整学习率。

### 9.3 `--lr`

- 只训练分类头：`3e-4`～`1e-3`。
- 全量微调：`1e-5`～`1e-4`。
- loss 发散：降低学习率。
- 训练准确率长期不变：检查标签，再测试更高学习率。

脚本使用 AdamW 和 CosineAnnealingLR。

### 9.4 `--weight-decay`

默认 `1e-2`，通过限制权重规模减少过拟合：

- 小数据、全量微调：可测试 `1e-3`、`1e-2`。
- 太大可能造成欠拟合。
- 最佳值必须看验证集，而非训练 loss。

### 9.5 `--freeze`

冻结 `model.features`，只训练分类器。代码还会让冻结的 features 进入 eval 模式，从而固定 BatchNorm 的 running mean/variance。

仅设置 `requires_grad=False` 而保留 BatchNorm 的 train 模式，仍会更新运行统计量，可能破坏预训练分布。这是迁移学习中常见但容易忽略的问题。

### 9.6 `--amp` / `--no-amp`

MPS 上默认为模型前向启用 FP16 autocast；当前实现不使用 CUDA GradScaler。

如果出现 NaN、特定算子精度异常或结果不稳定，可测试：

```bash
--no-amp
```

AMP 是否更快取决于模型、输入尺寸和 PyTorch/MPS 版本，必须实测。

### 9.7 `--workers`

macOS 先使用 `0`，稳定后测试 2 或 4：

- GPU/MPS 利用率低且 CPU 数据增强忙：增加 workers 可能有用。
- 小数据或图片已缓存：多进程启动成本可能更高。
- workers 大会增加内存和文件句柄压力。

## 10. checkpoint 内容和自定义推理

训练输出包含：

- `state_dict`。
- `classes` 类别顺序。
- 完整训练参数 `config`。

当前 [`infer.py`](../../mobilenet_v3/infer.py) 专门用于 ImageNet 预训练权重，不会自动加载这个自定义 checkpoint。自定义部署时需要：

1. 按 `config.variant` 重建 MobileNetV3。
2. 把最后一层改成 `len(classes)`。
3. 加载 `state_dict`。
4. 使用与训练一致的输入大小和 normalization。
5. 用 `classes[predicted_index]` 恢复业务类别名。

## 11. 分类评估指标

至少报告：

- Top-1 accuracy。
- 类别多时可报告 Top-5 accuracy。
- macro precision、recall、F1。
- 每类召回率和混淆矩阵。
- 不同输入分辨率下的延迟和精度。

类别不平衡时，overall accuracy 可能被多数类主导，macro F1 和每类召回率更有意义。

## 12. 性能优化实验

先跳过 compile：

```bash
uv run python -m mobilenet_v3.optimize \
  --runs 50 \
  --calib-dir data/calibration \
  --calib-samples 100 \
  --skip-compile
```

再测试 compile：

```bash
uv run python -m mobilenet_v3.optimize \
  --runs 50 \
  --calib-dir data/calibration \
  --calib-samples 100
```

脚本固定比较 ImageNet MobileNetV3-Large：

1. CPU FP32。
2. CPU 动态 int8。
3. CPU 静态 PTQ。
4. MPS FP32。
5. 可选 MPS `torch.compile`。

## 13. 动态量化与静态 PTQ

### 13.1 动态量化

动态量化只处理 `nn.Linear`。MobileNetV3 的主要计算在卷积特征骨干，所以它通常只缩小或加速分类头，整体收益有限。

### 13.2 静态 PTQ

静态 Post-Training Quantization 会量化卷积权重和激活。流程为：

```text
FP32 权重
  → 构建 quantizable MobileNetV3
  → fuse Conv/BN/Activation
  → prepare：插入 observer
  → 代表性图片校准
  → convert：生成 int8 模型
```

校准用于估计激活范围。如果校准图和生产图片分布不同，量化区间会失真，准确率可能明显下降。

推荐准备 100～500 张无标签、但具有代表性的图片，覆盖：

- 主要类别。
- 明暗变化。
- 相机和压缩质量。
- 常见背景。
- 实际输入预处理。

不传 `--calib-dir` 时，脚本只用一张示例图片做 smoke calibration。这只能证明流程可运行，不能证明 PTQ 精度。

## 14. 如何验证量化结果

当前脚本会比较示例图片的：

- FP32 Top-1。
- PTQ Top-1。
- 两组 logits 的余弦相似度。
- 模型大小。
- 延迟。

正式报告还必须在完整验证集重新计算 Top-1、macro F1 和混淆矩阵。单图 Top-1 相同不代表精度无损。

## 15. MPS、CPU int8 和 CoreML 怎么选

- MPS FP16/FP32：适合保留 PyTorch、需要灵活模型逻辑的场景。
- CPU int8：适合 CPU-only、低功耗或需要稳定线程控制的场景。
- CoreML：适合 Apple 端原生部署，可进一步利用 Apple 的部署栈，但需要单独验证转换算子和精度。

对于 batch=1 的小模型，MPS 的调度和数据传输开销可能抵消计算优势。选择必须基于端到端实测。

## 16. 常见故障

### 训练准确率很高，验证准确率很低

- 数据量太少或类别泄漏。
- 冻结策略不合适。
- 训练与验证预处理不同。
- 背景和类别高度相关，模型学到了背景捷径。

### 冻结后效果反而变差

目标域与 ImageNet 差异可能很大。先训练分类头稳定初始化，再用更低学习率解冻后几层或整个骨干。

### MPS 出现 NaN

先用 `--no-amp`，再降低学习率，并检查输入是否包含非有限值。

### PTQ 速度快但准确率明显下降

- 增加并改善校准集。
- 检查训练与量化预处理是否一致。
- 对敏感层保留 FP32。
- 考虑 Quantization-Aware Training。

## 17. 面试表达模板

> 我使用 torchvision 的 ImageNet 预训练 MobileNetV3 做业务分类迁移。小数据阶段先冻结 features，只训练新分类头，并把冻结骨干切到 eval 模式，避免 BatchNorm 统计量继续漂移。训练使用 ImageNet normalization、AdamW 和 cosine learning-rate schedule，在 MPS 上测试 AMP。部署优化时，我把动态量化作为对照，因为它只覆盖 Linear；真正针对卷积骨干的是带代表性校准集的静态 PTQ。最终同时比较验证集 Top-1、macro F1、模型大小和端到端延迟，而不是只看单张图片或模型文件大小。

常见追问：

- Depthwise convolution 为什么省计算：空间卷积不再同时完成全通道混合。
- 为什么冻结参数还要处理 BatchNorm：运行均值和方差不受 `requires_grad` 控制。
- 为什么动态量化收益小：MobileNet 的计算主要位于卷积而非 Linear。
- 校准集为什么不需要标签：observer 只估计激活分布，但精度评估仍需要标签。
- 分辨率如何选择：根据小目标细节、延迟和验证集精度做 Pareto 权衡。

## 18. 建议练习顺序

1. 使用 ImageNet 权重完成示例图片 Top-5 推理。
2. 用合成数据完成一次 MPS 冒烟训练。
3. 准备两到四类真实 ImageFolder 数据。
4. 对比冻结骨干与全量微调。
5. 扫描 160 和 224 分辨率。
6. 比较 CPU FP32、动态 int8、静态 PTQ 和 MPS。
7. 使用真实校准集和验证集重新评价 PTQ。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
