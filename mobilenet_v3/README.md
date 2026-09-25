# MobileNetV3 面试详解

## 1. 模型概况

| 项 | 说明 |
|---|---|
| 出处 | Google 2019《Searching for MobileNetV3》（NAS 搜索 + 手工微调） |
| 任务 | 移动端图像分类（可作检测/分割骨干） |
| 规模 | **small=2.5M / large=5.4M 参数**；ImageNet top-1：small=67.5% / large=75.2% |
| 输入 | 224×224 RGB（可缩到 160/128 换速度） |
| 本工程 | torchvision 预训练 + 自训迁移 + **静态 PTQ 完整实现** |

轻量级分类模型的三代演进：MobileNetV1（深度可分离卷积）→ V2（倒残差 + 线性瓶颈）→ V3（NAS 搜架构 + SE + h-swish）。面试时这条演进线是送分题。

## 2. 架构原理（逐个模块讲清）

### 2.1 深度可分离卷积 (DSC) —— V1 立足之本
标准卷积 `K×K×C_in×C_out` 拆成两步：
1. **Depthwise**：每个通道单独 K×K 卷积 → 计算量 1/C_out
2. **Pointwise**：1×1 跨通道混合

计算量从 `K²·C_in·C_out·H·W` 降到 `K²·C_in·H·W + C_in·C_out·H·W`，**约降 8~9 倍（K=3）**。

### 2.2 倒残差 + 线性瓶颈（Inverted Residual）—— V2
- 残差块窄-宽-窄（扩张 6 倍）→ 与 ResNet 的宽-窄-宽**相反**
- 原因：低维 ReLU 会丢信息，先把通道升维再 ReLU；**瓶颈层（输出）用线性**不激活，保住信息
- 输入输出通道少、中间宽 → 残差连接发生在低维空间，省参数

### 2.3 V3 的四个增量
1. **h-swish**：`x·relu6(x+3)/6`，swish 的高效近似。后 1/2 网络才用（浅层量化误差敏感区用 ReLU）
2. **SE 模块**（Squeeze-Excitation）：全局池化 → FC 降维 → FC 升维 → sigmoid 通道加权，插在残差分支内（在 DWC 后、PW 前，最省算子位置）
3. **NAS 搜出来的结构**：MnasNet 平台感知架构搜索，latency 直接进优化目标——精度/延迟 trade-off
4. **尾部重设计**：PW 后接 1×1 升维（延迟 -7ms, +1% 精度）、后段层宽度调整

### 2.4 为什么它量化友好？
- ReLU6 饱和激活天然限制激活范围 → int8 校准不易溢出
- DSC 的 depthwise 通道计算稀疏，量化后卷积（im2col/GEMM）收益大
- 浅层 h-swish 在 int8 下误差大 → 设计上只在深层用（这就是"量化感知的架构设计"案例）

## 3. 训练过程（本工程 `train.py`）

**范式：ImageNet 预训练 → 自定义类别迁移**
1. 加载 `MobileNet_V3_Small_Weights.DEFAULT`（也支持训练好的权重）
2. 替换 classifier 最后 Linear：`model.classifier[-1] = nn.Linear(1024, n_classes)`
3. 数据增强：RandomResizedCrop(scale 0.7~1.0)、HorizontalFlip；进阶：Mixup/CutMix/RandAugment
4. AdamW(3e-4) + Cosine + CrossEntropy
5. AMP fp16 autocast（MPS 原生支持）

**从零训练 ImageNet 注意**（区别于迁移）：需要 128 万张图/90 epoch，蒸馏版用 teacher（大模型 soft label）提升 1~3 个点——V3 论文本身就是蒸馏训练的。

**冻结微调**（`--freeze`）：小数据（每类 <50 图）必须冻 features 只训分类头，否则灾难性过拟合。

## 4. 参数调整

| 超参 | 推荐范围 | 经验 |
|---|---|---|
| learning rate | 迁移 3e-4 / 从零 0.1(SGD) | 分类头替换后需要 lr warmup 几个 epoch |
| weight decay | 1e-4~5e-4 | MobileNet 小容量对正则敏感，过大欠拟合 |
| label smoothing | 0.1 | 稳定提升 0.2~0.5% |
| imgsz | 128~224 | 端侧部署直接减输入分辨率（1/2 分辨率 ≈ 1/4 计算量） |
| epoch | 迁移 10~30 / 从零 90+ | 配合 early stopping |
| Mixup α | 0.2 | 从零训才用，小数据迁移慎用 |

**判断模型容量不够 vs 数据不够**：train acc 和 val acc 都低 → 容量问题（换 large / 加宽）；train 高 val 低 → 过拟合（加数据/增强/正则）。

## 5. 训练速度优化

- **AMP（本工程已实现）**：`torch.autocast(fp16)` 前向半精度，MPS 上 1.5~2x
- **channels_last 内存格式**：`model.to(memory_format=torch.channels_last)` + 输入同样转换，ARM/GPU 上卷积更快
- **冻结 + 小输入**：调参阶段 imgsz=160 冻结骨干，迭代速度可以快 5x
- **DataLoader**：num_workers>0 + persistent_workers + pin_memory（Mac 注意 spawn，Linux 全开）
- **torch.compile**（训练加速）：mode="max-autotune" 首次编译慢，但每步前反向最高省 30%
- **梯度累积**：小显存模拟大 batch

## 6. 推理优化（本工程 `optimize.py`，重点案例）

本工程完整实现了 **静态 PTQ 三步曲**，面试可全程白板：

```python
# ① 模块融合: Conv+BN+ReLU → 一个算子（浮点域做，量化前）
fused = copy.deepcopy(fp32); fused.fuse_model()
# ② 插桩 + 校准: QuantStub/DeQuantStub 定义量化边界；校准数据过一遍记录激活 min/max
wrapped = QuantWrapper(fused)
wrapped.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")  # ARM 后端
prepared = torch.ao.quantization.prepare(wrapped)
calibrate(prepared, calib_data)        # 100~500 张代表数据即可
# ③ 转换: 观察器换成真正的量化算子
qmodel = torch.ao.quantization.convert(prepared)
```

| 路线 | 特点 | 预期 |
|---|---|---|
| **动态量化** | 只 Linear 权重 int8，免校准 | small: 提速有限（模型以卷积为主） |
| **静态 PTQ（卷积也 int8）** | 需校准数据 + 融合 | 内存 9.7MB→2.5MB，CPU 提速 1.5~3x，top-1 掉 0.5~1% |
| **QAT 量化感知训练** | 训练期插伪量化节点 | 精度几乎无损（<0.3%），成本高 |
| **torch.compile** | Inductor kernel 融合 | 1.2~2x |
| **批量推理** | 多图拼 batch | 吞吐近线性 |

**校准数据怎么选**（高频追问）：必须是测试分布的代表样本（各目标类别均衡、覆盖亮度/场景多样性）；极端样本过多会拉宽量化区间掉精度，可用 KL 散度/percentile 校准替代 MinMax。

**速度不够时的决策树（背）**：先降输入分辨率（最便宜）→ 换 small 变体 → int8 量化 → 通道剪枝 → NAS 搜更小结构。

## 7. 面试 Q&A

**Q1: 为什么叫"倒残差"？与 ResNet 残差块的区别？**
A: ResNet：宽-窄-宽（1×1降维→3×3→1×1升维），残差连接在宽处。MobileNetV2：窄-宽-窄（1×1升维6x→DWC→1×1降维），残差连接在窄处。核心是低维空间做信息压缩（线性瓶颈不做激活，避免低维 ReLU 信息丢失），省参数且高效。

**Q2: DSC 为什么能省计算？有代价吗？**
A: 空间相关性与通道相关性分离建模。代价：depthwise 每通道单卷积核 → 算力低但 **访存/计算比高（memory-bound）**，在 GPU 上利用率低，实际提速在移动 CPU 才显著；且单通道滤波表达力弱，需要堆更多层。

**Q3: h-swish 为什么不完全用 swish？**
A: ① swish 求导有 exp 开销 ② 浅层激活值分布宽、int8 量化后误差大。V3 策略：浅层 ReLU（量化稳）+ 深层 h-swish（relu6 近似足够准）——精度与延迟/量化的平衡。

**Q4: PTQ 和 QAT 的选择？**
A: PTQ：免训练、分钟级、精度掉 1% 上下，优先试；QAT：插伪量化节点模拟量化噪声训练，精度几乎无损但要训练管线与算力。生产常见 PTQ 不达标 → QAT 兜底。

**Q5: 校准数据分布偏了会怎样？**
A: observer 统计的激活范围偏离真实 → 量化 scale 不准 → 严重时输出错乱（不止掉点）。方案：校准集覆盖部署分布；对离群点用 percentile(99.9%) 或 KL 散度校准。

**Q6: 怎么把 MobileNetV3 用到检测任务？**
A: 当骨干替换 SSD/YOLO-lite 检测头（如 SSDLite-MobileNetV3），DSC 检测头（普通卷积换 DSC）再砍 30% 计算量；或者作 FPN 特征源接分割头（DeepLabV3+）。 torchvision 官方有 SSDLite 与 FCOS 实现。
