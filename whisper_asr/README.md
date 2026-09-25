# Whisper（tiny / base / small）面试详解

建议先按 [`docs/tutorials/WHISPER_TUTORIAL.md`](../docs/tutorials/WHISPER_TUTORIAL.md) 完成训练、参数、推理和性能优化的完整实验，再使用本文复习原理与面试问答。

## 1. 模型概况

| 项 | 说明 |
|---|---|
| 出处 | OpenAI 2022《Robust Speech Recognition via Large-Scale Weak Supervision》 |
| 任务 | 多语言 ASR（语音→文本）+ 语种识别 + 翻译（X→English）+ 时间戳 |
| 训练数据 | 68 万小时互联网弱标注音频（比之前所有监督数据集之和还大一个量级） |
| 规模 | tiny=39M / base=74M / small=244M / medium=769M / large-v3=1.55B |
| 输入 | 30s 定长 log-Mel 频谱（80 维 × 3000 帧） |
| 输出 | 文本 token 序列（BPE，多语言词表 51865） |

**核心卖点**：超大数据 + 弱监督 + 无 N-gram 语言模型/发音词典，端到端 zero-shot 泛化到未见域（噪声、口音、远场）。

## 2. 架构原理（面试必讲）

```
音频(30s) --Mel频谱--> Encoder(Transformer) --中间嵌入--> Decoder(Transformer, 自回归)
                                                        |
                                          <start> <zh> < transcription tokens> <end>
```

- **Encoder**：2 层卷积（局部模式）+ N 层 Transformer。tiny=4 层/small=12 层，宽度 384/768
- **Decoder**：自回归生成。特殊 token 控制行为：`<|startoftranscript|>`、语言 token、任务 token（`transcribe`/`translate`）、`<|notimestamps|>`
- **为什么 30 秒定长**：训练简单（无需变长 batch），长音频靠滑窗/切分处理
- **幻觉问题**：自回归 LM 头会"脑补"重复文本，处理手段——温度回退、压缩比检测、no_speech 过滤

**tiny/base/small 关键超参**：

| | layers | width | heads | 参数 |
|---|---|---|---|---|
| tiny | 4 | 384 | 6 | 39M |
| base | 6 | 512 | 8 | 74M |
| small | 12 | 768 | 12 | 244M |

## 3. 训练过程（本工程 `train_finetune.py`）

**目标**：在垂直域（方言/术语/客服语音）上适配 whisper，而非从零训练（从零需数千 GPU 时）。

流程：
1. 数据准备：16k 音频 + 文本转写对，转成 `datasets` 格式
2. 特征：`WhisperProcessor` 提取 80 维 log-mel
3. 标签：tokenizer 编码文本，pad 时填 -100（CrossEntropy 忽略位）
4. 微调策略（小内存三板斧，M5 24GB 可用）：
   - **冻结 Encoder**：参数从 39M 全开降到只训 decoder，本工程 `--train_encoder` 开关对比
   - **减小 batch + 梯度累积**：等效大 batch
   - **梯度检查点**：显存不足时用重计算换激活内存（`--gradient-checkpointing`）
5. 评估：WER（词错误率）/ CER（中文字错误率）

**损失**：标准交叉熵（teacher forcing），无 CTC——Whisper 是纯 seq2seq。

## 4. 参数调整（调参经验）

| 超参 | 推荐值 | 说明 |
|---|---|---|
| learning_rate | 1e-5 ~ 6.4e-5 | 微调常用 1e-5 量级；过大毁预训练权重 |
| batch size | M5 上 1~4 起步 | 小显存用梯度累积获得更大的等效 batch |
| epochs | 2~5 | 弱监督基座易过拟合，小数据尤其早停 |
| warmup_ratio | 0.03~0.1 | 稳住前期训练 |
| max_length | 225 token | 官方标签截断长度 |
| 数据增强 | SpecAugment(时间/频域遮挡)、速度扰动 ±10% | 提升鲁棒性 |

**过拟合信号**：train loss 降但 WER 上升 → 加 SpecAugment / 减 epoch / 降 lr。

## 5. 训练速度优化

- **MPS 训练**：当前 Trainer 路径保持 fp32，避免把 CUDA GradScaler 的结论直接套到 MPS
- **冻结 encoder**：反向只过 decoder，约 40% 计算省下
- **数据管道**：`datasets.map` + `num_proc` 并行预提取特征并缓存（.arrow 列存），避免训练时现算 mel
- **DDP/ZeRO**：多卡时 `torchrun` + FSDP/ZeRO-3 切分 optimizer state；M5 单机则用 batch 堆叠 + pin_memory
- **梯度检查点**（gradient checkpointing）：用 ~30% 时间换 ~50% 显存，可开更大 batch 反而总吞吐更高

## 6. 推理优化（本工程 `optimize.py`）

| 手段 | 原理 | 实测预期 |
|---|---|---|
| **int8 动态量化** | Linear 权重 int8 存储计算，激活运行时量化 | 文件通常明显缩小；延迟与 WER 必须实测 |
| **torch.compile** | 图捕获、算子融合与内核生成 | MPS/自回归可能图断裂；首次编译单独计时 |
| **faster-whisper / CTranslate2** | 整图转 int8 + beam 剪枝 + KV 复用 | 4x 提速、内存减半（mac 仅 CPU，思路相同） |
| **Distil-Whisper** | 蒸馏：teacher=large-v3，student=small 结构 | 2x 参数小 5.8x、6x 快 |
| **批处理** | 多段音频拼 batch 过 encoder | 吞吐近线性提升 |
| **beam=1 (greedy)** | 解码从 beam5 降为贪心 | 解码时间 ~1/3，精度损失小 |

**工程链路范例**（背下来）：VAD 切片 → 重采样 16k → padding 到 30s 整数倍 → batch 过 encoder → greedy/beam 解码 → 时间戳对齐合并。

## 7. 面试 Q&A

**Q1: Whisper 为什么不用 CTC，而用 seq2seq？**
A: CTC 假设条件独立，无法直接建语言模型能力；Whisper 用 decoder 内置 LM，且时间戳/翻译/语种都统一成 token 生成，一个架构多任务。代价是自回归解码慢 + 有幻觉。

**Q2: WER 和 CER 区别？中文用哪个？**
A: WER 按词替换/删除/插入计错；中文无天然空格分词，用 CER（字符级）更稳定，且要做文本归一化（大小写、标点、数字格式）再算，否则虚高。

**Q3: 量化后精度掉了怎么办？**
A: 阶梯方案：① 回退 QAT（量化感知训练）② 只量化 FFN 保留 attention 高精度 ③ 混合 int8/int16 权重。动态量化对 Linear 为主的大模型通常掉点 <1% WER。

**Q4: 长音频如何处理？**
A: 先 VAD 切语音段，按 30s 窗口 padding 推理，token 级时间戳映射回原时间轴；注意窗口重叠区去重与 condition_on_previous 控制上下文。

**Q5: 微调 100 小时数据，冻结 encoder 还训全参？**
A: 先冻结 encoder 只训 decoder（保底）；若 WER 仍高说明声学域差距大（如强噪声），再解冻 encoder 用更小 lr（1e-6）继续。数据 <10h 时考虑 LoRA/Adapter 只插低秩矩阵。

**Q6: Whisper 面临什么安全/工程坑？**
A: ① 幻觉重复（beam+patience+温度回退缓解）② 30s 窗口边界切词（重叠切分）③ 多语种自动检测错误（显式传 language）④ 弱监督数据含偏见（脏数据风险）。
