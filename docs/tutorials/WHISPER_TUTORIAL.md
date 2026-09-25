# Whisper ASR 实战教程：训练、参数、推理与调优

本教程面向 Apple Silicon，默认实验环境为 MacBook Air M5、24GB 统一内存。目标不是只运行一次示例，而是理解 Whisper 的完整工程链路，并能在面试中解释每个选择。

对应代码：

- [`whisper_asr/train_finetune.py`](../../whisper_asr/train_finetune.py)：微调
- [`whisper_asr/infer.py`](../../whisper_asr/infer.py)：推理
- [`whisper_asr/optimize.py`](../../whisper_asr/optimize.py)：性能对比
- [`whisper_asr/README.md`](../../whisper_asr/README.md)：架构与面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 解释 Whisper 的 Encoder-Decoder 架构和自回归解码过程。
2. 在 M5 上完成一次可复现的微调实验。
3. 理解 batch、梯度累积、学习率、warmup、冻结 Encoder 和梯度检查点。
4. 使用预训练模型和微调模型转写自己的音频。
5. 公平比较 CPU FP32、CPU 动态 int8、MPS 和 `torch.compile`。
6. 用 WER、延迟、模型大小和内存，而不是单一指标评价方案。

## 2. 模型原理

Whisper 是 Encoder-Decoder Transformer：

```text
16kHz 音频
    ↓
80 维 Log-Mel 频谱
    ↓
Transformer Encoder：提取声学表征
    ↓
Transformer Decoder：自回归生成文本 token
    ↓
Tokenizer 解码为文本
```

Encoder 处理整段音频特征，Decoder 根据 Encoder 输出和已经生成的 token，逐步预测下一个 token：

\[
P(Y\mid X)=\prod_t P(y_t\mid y_{<t},X)
\]

这里的 `X` 是音频特征，`y_t` 是第 `t` 个文本 token。训练时使用正确的历史 token，也就是 teacher forcing；推理时只能使用模型自己已经生成的 token。因此，训练可以并行处理标签序列，而推理阶段的 Decoder 具有串行依赖。

### 2.1 为什么 Whisper 不是 CTC

CTC 通常假设帧级输出在给定输入后条件独立，再通过 blank 和路径合并得到文本。Whisper 使用自回归 Decoder，把语言建模、多语言识别、翻译和时间戳统一成 token 生成任务。

这样做的优点是语言建模能力强、任务统一；代价是推理速度较慢，并且在静音、强噪声或上下文不足时可能产生幻觉。

### 2.2 tiny、base、small 怎么选

| 模型 | 参数量 | M5 24GB 建议 |
|---|---:|---|
| tiny | 约 39M | 学习流程、快速调参、冒烟测试 |
| base | 约 74M | 速度和效果的折中 |
| small | 约 244M | 更重视精度，训练时使用小 batch |

第一次实验使用 `tiny`。确认数据、训练和评估都正确之后，再放大模型。不要在流程还未验证时直接使用 `small`。

## 3. 准备环境

进入仓库并按锁文件安装 Python 3.12 环境：

```bash
cd /Users/yu/Code/ml-models-cookbook
uv sync --python 3.12
```

先运行不需要下载大型模型的测试：

```bash
uv run pytest
uv run ruff check .
```

再验证 MPS、模型下载和音频处理链路：

```bash
uv run python -m whisper_asr.infer \
  --model tiny \
  --device mps \
  --language en
```

脚本会下载 JFK 示例音频。正常情况下会输出类似：

```text
[INFO] device=mps, model=openai/whisper-tiny
[RESULT] And so my fellow Americans ...
```

如果这一步失败，应先解决环境、网络或 MPS 问题，再开始训练。

## 4. 第一次训练：冒烟实验

先用极少样本验证整个训练链路：

```bash
uv run python -m whisper_asr.train_finetune \
  --model tiny \
  --epochs 1 \
  --train-samples 16 \
  --eval-samples 4 \
  --batch 2 \
  --grad-accum 2 \
  --lr 1e-5 \
  --output runs/whisper-smoke
```

这次实验不用于证明模型精度。它只验证：

- 数据能够下载和解码。
- 音频能够统一重采样到 16kHz。
- Log-Mel、attention mask 和文本标签形状正确。
- MPS 可以完成前向、反向和参数更新。
- 验证阶段可以生成文本并计算 WER。
- 模型和 Processor 能保存到磁盘。

最终模型位于：

```text
runs/whisper-smoke/final
```

### 4.1 数据如何进入模型

当前教学脚本使用 `PolyAI/minds14` 的 `en-US` 子集。数据流为：

```text
音频
  → 重采样为 16kHz
  → WhisperProcessor 提取 Log-Mel
  → input_features + attention_mask

转写文本
  → WhisperTokenizer
  → token ids
  → batch padding
  → padding 位置替换为 -100
```

标签中的 `-100` 会被交叉熵忽略，避免 padding 参与损失计算。

当前数据集只适合演示训练流程。正式项目必须换成目标业务数据，并准备可靠的训练集、验证集和测试集划分。

### 4.2 当前策略不是 LoRA

脚本默认采用“冻结 Encoder、训练 Decoder”：

```python
for parameter in model.model.encoder.parameters():
    parameter.requires_grad = False
```

它和 LoRA 的区别是：

- 冻结 Encoder：Decoder 的原始参数仍然直接更新。
- LoRA：原始参数冻结，只训练插入到特定线性层的低秩矩阵。
- 当前工程没有插入 LoRA Adapter，因此面试时不要称它为 LoRA。

## 5. 逐个理解训练参数

### 5.1 `--batch`

表示每次前向和反向实际送入设备的样本数。

| 模型 | M5 24GB 起步值 |
|---|---:|
| tiny | 2～4 |
| base | 1～2 |
| small | 1 |

Apple Silicon 使用统一内存，模型、激活、系统和其他应用共享内存。出现明显 swap 时，应降低 batch，而不是继续把内存占满。

### 5.2 `--grad-accum`

梯度累积在多次反向传播后才更新一次参数。单设备上的等效 batch 为：

\[
B_{effective}=B_{device}\times N_{accum}
\]

例如 `batch=2`、`grad-accum=4`，等效 batch 是 8。它能降低单步峰值内存，但不会减少总计算量，而且更新间隔会变长。

### 5.3 `--lr`

学习率控制每次参数更新的幅度：

- 冻结 Encoder：可从 `1e-5`～`3e-5` 开始。
- 全参数微调：可从 `1e-6`～`1e-5` 开始。
- loss 剧烈震荡或验证指标快速恶化：降低学习率。
- loss 几乎不动：先检查标签和数据，再考虑提高学习率。

微调时学习率过大可能快速破坏预训练表示，这通常称为 catastrophic forgetting。

### 5.4 `--epochs`

小数据集通常从 2～5 轮开始。典型过拟合信号是：

```text
train loss 持续下降
eval WER 先下降，随后升高
```

此时可减少 epoch、降低学习率、增加数据或使用更合理的数据增强。

### 5.5 `--warmup-ratio`

`warmup-ratio=0.05` 表示前 5% optimizer steps 逐渐提高学习率。warmup 可以避免训练初期的大梯度立即破坏预训练参数。数据少、全参数微调或等效 batch 较大时，可尝试 `0.05`～`0.1`。

### 5.6 `--train_encoder`

默认冻结 Encoder。增加这个开关后进行全参数微调：

```bash
--train_encoder
```

选择原则：

- 主要变化是专业术语、输出格式或表达方式：先冻结 Encoder。
- 口音、信道、噪声、麦克风或声学环境变化明显：考虑解冻 Encoder。
- 解冻后通常需要更低的学习率和更多内存。

### 5.7 `--gradient-checkpointing`

梯度检查点不保存部分中间激活，而是在反向传播时重新计算：

```bash
--gradient-checkpointing
```

它用额外计算时间换取更低的激活内存，适合 `base`、`small` 或全参数微调。tiny 且 Encoder 冻结时通常没有必要开启。

### 5.8 `--train-samples`、`--eval-samples` 和 `--seed`

- `train-samples`：本次使用的训练样本数。
- `eval-samples`：本次使用的验证样本数。
- `seed`：控制常见随机源，便于复现实验。

4 个验证样本只能检查流程，无法得到稳定的 WER。正式比较需要固定、更大的验证集，并避免用测试集调参。

## 6. 两组可比较的训练实验

### 6.1 实验 A：冻结 Encoder

```bash
uv run python -m whisper_asr.train_finetune \
  --model tiny \
  --epochs 3 \
  --train-samples 160 \
  --eval-samples 40 \
  --batch 2 \
  --grad-accum 4 \
  --lr 1e-5 \
  --warmup-ratio 0.05 \
  --seed 42 \
  --output runs/whisper-tiny-frozen
```

等效 batch 为 `2 × 4 = 8`。

### 6.2 实验 B：训练 Encoder 和 Decoder

```bash
uv run python -m whisper_asr.train_finetune \
  --model tiny \
  --train_encoder \
  --gradient-checkpointing \
  --epochs 3 \
  --train-samples 160 \
  --eval-samples 40 \
  --batch 1 \
  --grad-accum 8 \
  --lr 3e-6 \
  --warmup-ratio 0.1 \
  --seed 42 \
  --output runs/whisper-tiny-full
```

比较时至少记录：

| 指标 | 目的 |
|---|---|
| train/eval loss | 检查收敛和过拟合 |
| WER | 衡量识别错误率 |
| 单轮训练时间 | 衡量训练成本 |
| 峰值内存 | 判断设备可承受的配置 |
| 推理延迟 | 判断部署性能 |
| 输出案例 | 检查数字、专有名词和幻觉 |

WER 定义为：

\[
WER=\frac{S+D+I}{N}
\]

其中 `S` 是替换数，`D` 是删除数，`I` 是插入数，`N` 是参考文本的词数。WER 越低越好。中文通常更适合报告 CER，并且必须先统一大小写、标点和数字格式。

## 7. 使用预训练模型推理

### 7.1 英文示例

```bash
uv run python -m whisper_asr.infer \
  --model tiny \
  --audio /绝对路径/example.wav \
  --language en \
  --device mps \
  --max-new-tokens 128
```

### 7.2 中文示例

```bash
uv run python -m whisper_asr.infer \
  --model tiny \
  --audio /绝对路径/chinese.wav \
  --language zh \
  --device mps
```

### 7.3 自动识别语言

```bash
uv run python -m whisper_asr.infer \
  --model tiny \
  --audio /绝对路径/example.wav \
  --language auto \
  --device mps
```

已知语言时建议显式指定，避免短音频或噪声音频的语言识别错误。

`max-new-tokens` 控制最多生成多少文本 token。设置过小会截断，设置过大则会增加解码时间，并可能让异常音频产生重复文本。短音频通常从 64～128 开始。

## 8. 使用微调模型推理

训练输出目录同时保存模型和 Processor，因此可以直接传给 `--model`：

```bash
uv run python -m whisper_asr.infer \
  --model runs/whisper-tiny-frozen/final \
  --audio /绝对路径/test.wav \
  --language en \
  --device mps \
  --max-new-tokens 128
```

对同一批测试音频分别运行预训练模型和微调模型，才能判断微调是否真正改善目标域。不要只展示训练 loss。

## 9. 性能优化实验

先跳过编译，比较稳定路径：

```bash
uv run python -m whisper_asr.optimize \
  --model tiny \
  --runs 20 \
  --max-new-tokens 20 \
  --skip-compile
```

脚本比较：

1. CPU FP32。
2. CPU 动态 int8。
3. MPS FP32。
4. 模型权重大小。
5. FP32 与 int8 的输出文本。

再测试 `torch.compile`：

```bash
uv run python -m whisper_asr.optimize \
  --model tiny \
  --runs 20 \
  --max-new-tokens 20
```

第一次编译可能需要较长时间。编译时间属于启动成本，不应混入稳定推理延迟；但对于只调用一次的桌面程序，启动成本仍然是真实用户成本，也应单独记录。

### 9.1 动态 int8 做了什么

当前脚本只动态量化 `nn.Linear`：

```python
torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8,
)
```

Linear 权重由 FP32 变成 int8，激活在运行时量化。它无需校准集，适合作为低成本基线，但并没有把整个模型的所有算子都变成 int8。

### 9.2 为什么 int8 可能更慢

Whisper 的端到端推理包含：

- Encoder 计算。
- Python 层生成循环。
- 一次一个 token 的小矩阵运算。
- Softmax、LayerNorm 和其他未量化算子。
- 动态量化与反量化开销。

因此，动态 int8 往往能缩小权重，但不保证降低短音频的端到端延迟。本仓库在 M5 冒烟测试中也观察到了“模型变小但动态 int8 更慢”的结果。

正确结论是：量化是候选方案，不是必然加速方案。必须在目标硬件上同时测量延迟、内存、模型大小和任务精度。

### 9.3 公平 benchmark 的要求

比较不同后端时，应保持：

- 相同模型和输入音频。
- 相同预处理。
- 相同 `max-new-tokens` 和解码策略。
- 相同后处理。
- 独立 warmup。
- MPS 计时前后进行同步。
- 重复多次并记录 mean、p50、p95，而不是只取最快一次。

当前脚本完成 warmup、设备同步和输出一致性检查。正式报告还应加入真实验证集 WER 和峰值内存。

## 10. 下一步优化方向

### 10.1 先缩小问题，而不是只优化模型

生产 ASR 常见链路是：

```text
长音频
  → VAD 去除静音并切段
  → 统一为 16kHz 单声道
  → 多段组成 batch 经过 Encoder
  → greedy 或 beam search 解码
  → 按时间戳合并文本
```

VAD 可以避免把大量静音送给 Whisper，也能减少静音幻觉。

### 10.2 模型和解码策略

- 从 `small` 换成 `base` 或 `tiny`，通常是最直接的速度优化。
- greedy 解码比 beam search 快，但需要测量 WER 差异。
- 限制 `max-new-tokens` 可以减少异常长解码。
- 多条短音频可以批量处理 Encoder，提高吞吐。
- 延迟敏感和吞吐敏感是不同目标，不能用一个数字同时代表。

### 10.3 数据侧优化

- 预先提取并缓存 Log-Mel，减少每个 epoch 的 CPU 预处理。
- 按音频长度分桶，减少 batch 内 padding。
- 将极长音频提前切段。
- 保持训练和部署时的采样率、声道处理和文本归一化一致。

## 11. 常见故障

### MPS 内存不足

依次尝试：

1. 降低 `--batch`。
2. 增大 `--grad-accum` 保持等效 batch。
3. 保持 Encoder 冻结。
4. 开启 `--gradient-checkpointing`。
5. 从 `small` 降到 `base` 或 `tiny`。

### WER 很高

检查：

- 音频与转写是否一一对应。
- 采样率是否正确。
- `language` 和 `task` 是否正确。
- 文本是否做了一致的大小写、标点和数字归一化。
- 验证集是否与训练集分布不同。
- 是否使用过小的验证集得出了不稳定结论。

### 训练 loss 正常但生成文本异常

检查 Decoder 起始 token、语言 token、任务 token、padding 是否替换为 `-100`，以及保存和加载时是否同时保存了 Processor。

### `torch.compile` 没有变快

自回归生成包含动态控制流，可能发生 graph break；编译成本也可能大于收益。应把编译视作实验选项，而不是默认结论。

## 12. 面试表达模板

可以按“任务—约束—方案—实验—结论”回答：

> 我在 Apple Silicon 上完成了 Whisper tiny 的领域微调和推理优化。训练阶段将音频统一重采样到 16kHz，通过 WhisperProcessor 提取 Log-Mel，并使用 teacher forcing 和交叉熵训练。针对 24GB 统一内存，我先冻结 Encoder，只训练 Decoder，并使用小 batch 和梯度累积控制峰值内存，评估指标使用 WER。推理阶段对比了 CPU FP32、动态 int8、MPS 和 torch.compile。实验发现动态量化虽然缩小了模型，但短序列自回归解码不一定更快，因此最终方案必须根据目标硬件上的端到端延迟、内存和 WER 共同决定。

面试官继续追问时，需要主动说明：

- 当前冻结 Encoder 的实现不是 LoRA。
- 领域变化只有文本侧时优先训练 Decoder；声学域变化明显时再解冻 Encoder。
- 中文通常报告 CER，英文通常报告 WER。
- 量化后不仅检查输出文本，还应在完整验证集上重新计算 WER。
- 长音频需要 VAD、重叠切窗和时间戳合并，不能直接无限延长输入。

## 13. 建议练习顺序

1. 用预训练 tiny 转写 JFK 示例。
2. 完成 16/4 样本的冒烟训练。
3. 用保存的模型重新推理。
4. 完成冻结 Encoder 与全参数训练的对照实验。
5. 记录 loss、WER、时间和峰值内存。
6. 比较 CPU FP32、CPU int8 和 MPS。
7. 换成一段自己的中文或英文音频。
8. 用实验模板写出一页结论，解释为什么最终选择某个模型和后端。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
