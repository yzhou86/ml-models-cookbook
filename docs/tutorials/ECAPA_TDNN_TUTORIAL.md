# ECAPA-TDNN 声纹实战教程：训练、参数、推理与调优

本教程介绍 ECAPA-TDNN 的说话人表征、验证、分类适配和性能优化，默认环境为 MacBook Air M5、24GB 统一内存。

对应代码：

- [`ecapa_tdnn/train.py`](../../ecapa_tdnn/train.py)：冻结 ECAPA 后训练业务分类头
- [`ecapa_tdnn/infer.py`](../../ecapa_tdnn/infer.py)：说话人验证
- [`ecapa_tdnn/optimize.py`](../../ecapa_tdnn/optimize.py)：FP32、动态 int8 和 TorchScript 对比
- [`ecapa_tdnn/README.md`](../../ecapa_tdnn/README.md)：架构与面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 区分说话人验证、识别和闭集分类。
2. 解释 ECAPA-TDNN 如何把变长语音映射成固定维度 embedding。
3. 使用预训练模型计算两段音频的余弦相似度。
4. 使用 AAM-Softmax 训练业务说话人分类头。
5. 理解 margin、scale、阈值以及 EER、FAR、FRR。
6. 判断动态量化为何对卷积占主导的 ECAPA 收益有限。

## 2. 先区分三个任务

### 2.1 说话人验证

输入两段音频，回答是否属于同一人：

```text
audio A → ECAPA → embedding A ┐
                              ├→ cosine → threshold → same/different
audio B → ECAPA → embedding B ┘
```

这是 [`infer.py`](../../ecapa_tdnn/infer.py) 完成的任务。

### 2.2 闭集说话人分类

输入一段音频，在训练时见过的固定说话人集合中选择身份。例如公司内部的 100 个注册用户。

这是 [`train.py`](../../ecapa_tdnn/train.py) 训练的业务分类头所解决的任务。

### 2.3 开集说话人识别

先与声纹库中的 embedding 做检索，再根据阈值决定是否为已注册用户。它结合了最近邻搜索和拒识机制，不能只用分类准确率评价。

当前训练脚本保存的 `finetune.pt` 不会被验证脚本自动加载。原因是验证脚本直接比较预训练 ECAPA embedding，而训练脚本额外学习的是闭集分类头。面试时必须说明这两个路径的任务边界。

## 3. ECAPA-TDNN 原理

ECAPA 是 Emphasized Channel Attention, Propagation and Aggregation in TDNN。简化链路如下：

```text
16kHz 波形
  → FBank 特征
  → TDNN / Res2Net 多尺度时序卷积
  → SE 通道注意力
  → 多层特征聚合
  → attentive statistics pooling
  → 192 维说话人 embedding
```

### 3.1 为什么使用时序卷积

说话人特征分布在音色、共振峰、发音习惯和韵律中。TDNN 使用不同时间上下文的卷积建模局部和中期模式，比逐帧分类更适合声纹。

### 3.2 Attentive Statistics Pooling

输入语音长度不固定，但 embedding 维度必须固定。统计池化计算带权均值和标准差：

\[
\mu=\sum_t\alpha_t h_t
\]

\[
\sigma=\sqrt{\sum_t\alpha_t(h_t-\mu)^2}
\]

注意力权重 `α` 让模型强调更有说话人区分度的帧，弱化静音和无效片段。

## 4. 第一次推理

默认命令会用同一段 JFK 音频自比较，相似度应接近 1：

```bash
cd /Users/yu/Code/ml-models-cookbook
uv run python -m ecapa_tdnn.infer
```

第一次运行会下载 SpeechBrain 的 `spkrec-ecapa-voxceleb` 预训练模型。

比较两段自己的音频：

```bash
uv run python -m ecapa_tdnn.infer \
  --a /绝对路径/speaker_a.wav \
  --b /绝对路径/speaker_b.wav \
  --threshold 0.75 \
  --device cpu
```

默认使用 CPU，因为 SpeechBrain 部分算子在 MPS 上的覆盖和稳定性不如标准 PyTorch 模块。当前脚本允许显式测试 `--device mps`，但生产选择应以完整数据和稳定性验证为准。

## 5. 理解 embedding 和余弦相似度

模型输出 192 维 embedding。两向量的余弦相似度为：

\[
cos(a,b)=\frac{a\cdot b}{\|a\|\|b\|}
\]

相似度越高，模型越倾向于认为两段音频属于同一人。但阈值不是模型的全球固定常数，它会受到以下因素影响：

- 录音设备和信道。
- 音频长度。
- 背景噪声和混响。
- 语言、口音和情绪。
- 是否存在重放、变声或合成攻击。

脚本默认阈值 `0.75` 只用于演示。生产阈值必须在目标域验证集上标定。

## 6. 准备分类适配数据

目录结构：

```text
data/spk/
├── speaker_001/
│   ├── a.wav
│   ├── b.wav
│   └── c.flac
├── speaker_002/
│   ├── a.wav
│   └── b.wav
└── ...
```

要求：

- 至少两个说话人目录。
- 每个目录包含该说话人的 wav 或 flac。
- 最好统一为单声道 16kHz。
- 训练、验证和测试应按录音会话划分，而不是随机切割同一长录音。
- 每个人应覆盖不同文本、设备、距离和噪声。

## 7. 第一次训练：业务分类头

```bash
uv run python -m ecapa_tdnn.train \
  --data data/spk \
  --epochs 10 \
  --batch 64 \
  --lr 1e-3 \
  --margin 0.2 \
  --scale 30 \
  --device auto \
  --seed 42 \
  --out runs/ecapa-classifier
```

训练过程分成两阶段：

1. 预训练 ECAPA 固定在 CPU，对每个音频只计算一次 192 维 embedding。
2. 缓存全部 embedding，然后在 MPS 或 CPU 上训练轻量 MLP 和 AAM-Softmax 分类权重。

输出文件为：

```text
runs/ecapa-classifier/finetune.pt
```

其中保存：

- 业务 MLP head。
- AAM 分类权重。
- 说话人类别顺序。
- margin 和 scale 配置。

## 8. 为什么先缓存 embedding

ECAPA 骨干在此方案中完全冻结，同一音频在每个 epoch 得到的 embedding 不会变化。如果每轮都重新跑骨干，会重复消耗绝大多数时间。

预计算后的复杂度从：

```text
epochs × 所有音频 × ECAPA forward
```

变为：

```text
所有音频 × ECAPA forward
+ epochs × 轻量分类头 forward/backward
```

这是典型的冻结特征提取优化。但它也意味着无法进行波形级随机增强，因为每个 epoch 使用的 embedding 已固定。如果需要增强或微调骨干，就不能简单缓存一次。

## 9. AAM-Softmax 原理

普通 Softmax 只要求样本被正确分类，不直接约束 embedding 的角度间隔。AAM-Softmax 对目标类别加入角度 margin：

\[
logit_y=s\cdot cos(\theta_y+m)
\]

非目标类别保持：

\[
logit_j=s\cdot cos(\theta_j)
\]

这样会迫使同一说话人的 embedding 更紧，不同说话人之间具有更大的角度间隔。

代码没有直接计算 `acos`，而是利用：

\[
cos(\theta+m)=cos\theta cos m-sin\theta sin m
\]

这通常比频繁调用 `acos` 更稳定，并对超过单调区间的困难样本做了修正。

## 10. 训练参数怎么调

### 10.1 `--margin`

默认 `0.2`。margin 越大，分类边界要求越严格：

- 太小：embedding 区分度提升有限。
- 太大：早期训练困难，loss 高且不稳定。
- 小数据适配可从 0.1、0.2、0.3 扫描。

margin 必须和 scale、数据难度共同调节。

### 10.2 `--scale`

归一化后的余弦值只在 `[-1, 1]`，直接送入 Softmax 时梯度可能太弱。scale 把 logits 放大，默认 `30`。

- scale 太小：收敛慢，分类边界不明显。
- scale 太大：Softmax 过于尖锐，困难样本梯度可能不稳。
- 可以测试 16、30、32、48。

### 10.3 `--batch`

当前训练阶段只处理 192 维 embedding，内存开销很小，M5 24GB 可从 64 开始，并测试 128 或 256。

但 batch 大小不能解决类别采样不均衡。真实声纹训练通常使用每批 `N` 个说话人、每人 `M` 条音频的 speaker-balanced sampler。

### 10.4 `--lr` 和 `--epochs`

- 当前仅训练轻量头，可从 `1e-3` 开始。
- loss 震荡可降到 `3e-4`。
- epoch 根据独立验证集指标决定，不要只看训练准确率。

### 10.5 当前训练脚本的边界

当前 DataLoader 对所有文件普通随机打乱，且没有内置验证集。这适合教学和闭集适配基线，但正式声纹系统应增加：

- speaker-balanced sampling。
- 每个说话人的 train/validation/test 会话隔离。
- 波形增强和噪声混合。
- 验证 trial pair 列表。
- EER、minDCF、TAR@FAR 等指标。

## 11. 声纹评估指标

### 11.1 FAR 和 FRR

- FAR：不同人的样本被错误接受为同一人。
- FRR：同一人的样本被错误拒绝。

阈值升高通常降低 FAR、提高 FRR；阈值降低则相反。

### 11.2 EER

FAR 与 FRR 相等处的错误率称为 Equal Error Rate。EER 越低通常表示整体区分能力越好，但生产系统最终仍需按业务风险选择工作阈值。

### 11.3 TAR@FAR

高安全场景更常报告固定 FAR 下的真实接受率，例如 `TAR@FAR=1e-4`。这比一个演示阈值更有业务意义。

### 11.4 分类准确率为什么不够

训练准确率只衡量已知类别的分类，不等同于对未见说话人的验证能力。声纹 embedding 的核心评估必须基于同人和异人 pair。

## 12. 性能优化实验

```bash
uv run python -m ecapa_tdnn.optimize --runs 50
```

脚本会：

1. 对波形提取与预训练模型一致的 FBank 和归一化特征。
2. 单独取 ECAPA embedding 骨干做基准。
3. 比较 CPU FP32 与 Linear 动态 int8。
4. 尝试 TorchScript，失败时安全跳过。
5. 比较量化前后 embedding 的余弦相似度。

### 12.1 为什么动态 int8 收益有限

动态量化只覆盖 `nn.Linear`，而 ECAPA 的主要计算位于 Conv1d、Res2Net 和注意力统计模块。因此：

- 大部分权重和计算仍是 FP32。
- 模型大小可能几乎不变。
- 量化调度开销可能抵消 Linear 的加速。

这正是需要通过模型算子构成选择优化方案的案例：Transformer 的 Linear 比例高，动态量化更可能受益；卷积骨干则通常需要静态 PTQ、QAT 或专用推理引擎。

### 12.2 输出相似不等于系统精度不变

单条音频上 FP32/int8 embedding 余弦接近 1，只能证明该样本的输出接近。正式结论必须重新计算整个验证集的 EER、TAR@FAR 和阈值。

## 13. 进一步优化

### 13.1 输入裁剪

长音频可先用 VAD 去除静音，再限制最大时长。极短音频则可能缺少足够的说话人信息，需要设定最小时长。

### 13.2 批量提取 embedding

注册或离线建库时，可以把相近长度的音频组成 batch，减少 padding 并提高吞吐。

### 13.3 声纹库检索

embedding 应先 L2 归一化。小规模声纹库可直接矩阵乘法，大规模可使用向量索引，但必须保留拒识阈值。

### 13.4 分数归一化

跨域环境可考虑 cohort-based score normalization，例如 Z-norm、T-norm 或 S-norm。它们需要额外的背景说话人集合。

## 14. 常见故障

### 同一人的分数仍然很低

- 音频过短或静音过多。
- 两段音频的设备、噪声或距离差异很大。
- 出现多人同时说话。
- 采样率或音频解码异常。
- 直接使用了不适合目标域的阈值。

### 不同人的分数过高

- 阈值未经目标数据标定。
- 音频中包含相同背景声或重复播放内容。
- 注册音频质量差。
- 测试集太小，或者 pair 构造存在数据泄漏。

### MPS 推理失败

先改用默认 CPU：

```bash
uv run python -m ecapa_tdnn.infer --device cpu
```

SpeechBrain 模型由多个特征和归一化模块组成，MPS 支持情况可能随版本变化。不要把单个算子的支持等同于整条链路可稳定部署。

## 15. 面试表达模板

> 我使用 SpeechBrain 的预训练 ECAPA-TDNN 提取 192 维声纹 embedding，通过 L2 归一化后的余弦相似度完成说话人验证。业务适配时先冻结 ECAPA 骨干并预计算 embedding，再训练轻量 MLP 和 AAM-Softmax 分类头，从而避免每个 epoch 重复运行骨干。AAM-Softmax 在目标类别上加入角度 margin，使类内更紧、类间更分离。阈值不是拍脑袋设定，而是在目标域 trial pairs 上根据 FAR、FRR、EER 或 TAR@FAR 标定。性能优化中我验证了动态 int8，但 ECAPA 以卷积为主，Linear 量化收益有限，因此保留了真实对照结果。

常见追问：

- 为什么不能只看分类准确率：闭集准确率不代表未见说话人的验证能力。
- 为什么缓存 embedding：骨干冻结后输出不随 epoch 变化。
- 缓存的代价是什么：无法每轮做新的波形增强，也无法更新骨干。
- margin 和 scale 各做什么：margin 拉开角度边界，scale 提供合适的 Softmax 梯度尺度。
- 为什么阈值要重标定：设备、噪声、时长和攻击类型都会改变分数分布。

## 16. 建议练习顺序

1. 用同一音频自比较，确认相似度接近 1。
2. 准备同人和异人音频，观察相似度分布。
3. 准备两个以上说话人的目录，训练业务分类头。
4. 扫描 margin 0.1、0.2、0.3。
5. 构造验证 pairs，计算 FAR、FRR 和 EER。
6. 运行动态 int8 对比，解释为什么收益有限。
7. 记录 CPU 线程、音频长度和 batch 对吞吐的影响。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
