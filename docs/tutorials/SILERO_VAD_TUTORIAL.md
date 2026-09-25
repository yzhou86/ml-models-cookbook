# Silero VAD 实战教程：训练、参数、推理与调优

本教程介绍语音活动检测（Voice Activity Detection，VAD）的完整工程链路，默认环境为 MacBook Air M5、24GB 统一内存。

对应代码：

- [`silero_vad/train_custom_vad.py`](../../silero_vad/train_custom_vad.py)：自定义 TinyVAD 训练
- [`silero_vad/infer.py`](../../silero_vad/infer.py)：官方 Silero JIT 推理和切段
- [`silero_vad/optimize.py`](../../silero_vad/optimize.py)：JITScript 与 ONNX Runtime 对比
- [`silero_vad/README.md`](../../silero_vad/README.md)：原理和面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 解释 VAD 和简单能量阈值的区别。
2. 理解帧级概率、迟滞阈值、最短语音和静音 hangover。
3. 使用官方 Silero 模型切分一段长音频。
4. 使用语音和噪声数据训练一个轻量 TinyVAD。
5. 调整 threshold、窗口、片段长度和 CNN 容量。
6. 公平比较 JITScript 与 ONNX Runtime。

## 2. 先区分两个模型

本模块包含两条相互独立的路径：

| 路径 | 用途 | 脚本 |
|---|---|---|
| 官方 Silero VAD | 直接部署高质量预训练 VAD | `infer.py`、`optimize.py` |
| 自定义 TinyVAD | 学习训练流程或适配特殊业务域 | `train_custom_vad.py` |

`train_custom_vad.py` 训练出的 `tiny_vad.pt` 不会被 `infer.py` 自动加载，因为 `infer.py` 使用的是官方 Silero JIT 模型。自定义模型如果要部署，还需要实现对应的流式特征提取、阈值后处理和模型加载代码。

这是面试中必须说清楚的工程边界。

## 3. VAD 的基本原理

VAD 输入音频窗口，输出该窗口属于语音的概率：

```text
连续音频
  → 固定窗口
  → 模型输出 speech probability
  → 阈值判定
  → 时间平滑和短段过滤
  → [(start, end), ...]
```

官方 Silero 路径使用 16kHz 音频和 512 样本窗口：

\[
512 / 16000 = 0.032s = 32ms
\]

模型内部具有状态，因此前一帧的上下文会影响后一帧。处理新音频或新会话前必须重置状态。

### 3.1 为什么不用能量阈值

能量阈值只能判断声音大小，无法可靠区分人声、风声、键盘声或背景音乐。神经网络 VAD 学习的是人声模式，对噪声环境通常更稳健。

能量法仍有价值：它计算成本极低，可以作为神经网络 VAD 前的粗筛选，但不能代替真实评估。

## 4. 先运行官方 Silero 推理

默认使用 JFK 示例音频：

```bash
cd /Users/yu/Code/ml-models-cookbook
uv run python -m silero_vad.infer
```

程序将：

1. 下载经过固定提交和 SHA-256 校验的官方 JIT 模型。
2. 把音频转换为单声道 16kHz。
3. 每 512 样本执行一次推理。
4. 将逐窗概率合并成语音区间。
5. 把前三个语音片段保存到 `samples/vad_seg*.wav`。

处理自己的音频：

```bash
uv run python -m silero_vad.infer \
  --audio /绝对路径/meeting.wav \
  --threshold 0.5 \
  --min-speech-ms 100 \
  --min-silence-ms 100
```

## 5. 推理参数怎么调

### 5.1 `--threshold`

语音开始阈值，默认 `0.5`。代码内部使用迟滞策略：

- 概率达到 `threshold`：进入语音状态。
- 概率低于 `threshold - 0.15`：开始统计静音。

例如 `threshold=0.5` 时，进入阈值是 0.5，退出候选阈值是 0.35。两个阈值不相同可以防止概率在边界附近反复切换状态。

调参规律：

- 误检多：提高到 `0.6`～`0.7`。
- 漏掉轻声或远场语音：降低到 `0.35`～`0.45`。
- 不要只看几段音频，应在标注验证集上画 precision-recall 曲线。

### 5.2 `--min-speech-ms`

最短语音段，默认 100ms。短于这个长度的候选语音会被丢弃。

- 增大：减少敲击声、咳嗽等短误检。
- 减小：保留“嗯”“啊”等极短语音，但误检可能增加。

### 5.3 `--min-silence-ms`

语音结束前需要持续多久的静音，默认 100ms。它也叫 hangover：

- 增大：减少一句话被切成很多小段，但输出等待时间增加。
- 减小：切段响应更快，但容易切断词语或短暂停顿。

实时字幕可从 300～500ms 开始；离线 ASR 可以更长，再结合最大段长切分。

## 6. 训练前理解 TinyVAD

自定义训练模型使用：

```text
16kHz 波形
  → 40 维 Log-Mel，25ms 窗口，10ms 帧移
  → 4 层空洞 Conv1d
  → 每一帧一个 logit
  → BCEWithLogitsLoss
```

张量形状为：

```text
输入  (B, T, 40)
卷积  (B, C, T)
输出  (B, T)
标签  (B, T)
```

四层卷积的 dilation 为 1、2、4、8，kernel size 为 5。理论感受野约为 61 帧，即约 610ms 的 Log-Mel 上下文。空洞卷积在不显著增加参数量的情况下扩大时间感受野。

## 7. 第一次训练：合成数据冒烟实验

不传数据目录时，脚本会生成合成语音样波形和白噪声：

```bash
uv run python -m silero_vad.train_custom_vad \
  --epochs 2 \
  --batch 16 \
  --channels 64 \
  --samples-per-class 32 \
  --clip-seconds 2 \
  --out runs/tiny-vad-smoke.pt
```

这只能验证：

- Log-Mel 特征可以生成。
- 输入输出的时间长度一致。
- BCE 损失可以反向传播。
- MPS 训练和保存路径正常。

合成数据不能证明真实 VAD 精度。

## 8. 准备真实训练数据

目录结构：

```text
data/vad/
├── speech/
│   ├── speaker_001.wav
│   ├── speaker_002.flac
│   └── ...
└── noise/
    ├── office.wav
    ├── keyboard.wav
    └── ...
```

开始训练：

```bash
uv run python -m silero_vad.train_custom_vad \
  --speech-dir data/vad/speech \
  --noise-dir data/vad/noise \
  --epochs 20 \
  --batch 16 \
  --lr 3e-4 \
  --channels 64 \
  --samples-per-class 256 \
  --clip-seconds 2 \
  --seed 42 \
  --out runs/tiny-vad.pt
```

当前教学数据构造有一个重要限制：一条 speech 文件的所有帧都标为 1，一条 noise 文件的所有帧都标为 0。这适合纯语音与纯噪声分类演示，但不等价于生产级帧标注。

生产训练应使用包含语音边界的混合音频，并为每一帧提供 0/1 标注；还应加入不同 SNR 的语音噪声混合、混响、远场和设备响应。

## 9. 训练参数怎么调

### 9.1 `--channels`

控制四层卷积的通道数，是主要容量参数：

| channels | 特点 |
|---:|---|
| 32 | 更小更快，适合嵌入式基线 |
| 64 | 默认折中，约 75K 参数 |
| 96/128 | 容量更大，需检查是否真正改善 F1 |

通道数翻倍会显著增加中间卷积的参数和计算量。不要在数据很少时盲目扩大模型。

### 9.2 `--clip-seconds`

控制每个训练样本裁剪或补齐的时长：

- 太短：上下文单一，难覆盖真实变化。
- 太长：训练内存和特征预计算成本增加。
- 当前网络按帧训练，1～3 秒通常足以做实验。

### 9.3 `--samples-per-class`

每类构建多少个训练片段。真实数据池最多读取每类前 50 个文件，但可通过重复文件随机裁剪得到更多片段。

重复裁剪不等于增加独立说话人和噪声类型。正式数据集应优先增加覆盖度。

### 9.4 `--lr`、`--batch` 和 `--epochs`

- `lr`：从 `3e-4` 开始；loss 震荡可降到 `1e-4`。
- `batch`：M5 24GB 可从 16～64 扫描。
- `epochs`：先看验证集 F1，再决定是否继续，不能只看训练 loss。

当前脚本没有划分验证集，正式实验应补充按录音来源划分的 train/validation/test，避免同一录音的不同裁剪同时出现在训练和验证中。

## 10. VAD 评估指标

至少报告：

- 帧级 precision、recall、F1。
- False Alarm Rate：非语音被判成语音。
- Miss Rate：真实语音被漏掉。
- 段级边界误差或 segment IoU。
- ASR 下游 WER 和节省的计算量。

阈值不是模型固有常数，而是业务成本的选择：

- ASR 前置 VAD 通常更重视 recall，避免永久丢失语音。
- 静音唤醒或计费场景可能更重视 precision。

## 11. JITScript 与 ONNX Runtime 优化

运行基准：

```bash
uv run python -m silero_vad.optimize \
  --runs 50 \
  --threads 2
```

继续测试线程数：

```bash
uv run python -m silero_vad.optimize --runs 50 --threads 1
uv run python -m silero_vad.optimize --runs 50 --threads 2
uv run python -m silero_vad.optimize --runs 50 --threads 4
```

比较对象都是对整段示例音频逐个窗口处理：

- JITScript：调用 PyTorch JIT 模型，模型内部保存状态。
- ONNX Runtime：显式传入和接收形状为 `(2, B, 128)` 的状态。

测试线程数时不要假设更多线程一定更快。VAD 单次计算很小，过多线程可能增加调度成本，也会影响同时运行的 ASR。

### 11.1 为什么不优先做 int8

Silero 本身只有约 2MB，且包含循环状态结构。量化收益可能小于转换、兼容性和精度验证成本。因此本工程优先比较 JIT 与 ONNX，并优化窗口调度和多流并发。

这体现了一个重要原则：先找到真实瓶颈，再选择优化技术。

## 12. 生产优化思路

### 12.1 流式状态隔离

每个通话或音频流都必须拥有独立状态。把 A 用户的 state 用到 B 用户上会造成上下文污染。

### 12.2 避免重复内存分配

可以复用 512 样本输入缓冲区和 state 数组，减少高并发服务中的分配压力。

### 12.3 多路 batch

多个实时会话可以按时间片组成 batch，但需要同时维护每个会话的状态和结束信号。batch 可以提高吞吐，却可能增加等待延迟。

### 12.4 和 ASR 联合调参

VAD 本身 F1 最高的参数，不一定让最终 ASR WER 最低。边界过紧会切掉音素，边界过松会增加静音和幻觉。最终应联合评估：

```text
VAD 参数
  → 切分后的片段
  → Whisper 推理
  → WER、延迟和计算量
```

## 13. 常见故障

### 音频完全检测不到语音

- 确认文件不是空文件。
- 确认已转换为单声道 16kHz。
- 降低 `--threshold`。
- 检查声音幅度和是否存在严重削波。

### 一句话被切成很多段

- 增大 `--min-silence-ms`。
- 适当降低 threshold。
- 对输出区间增加边界 padding。

### 背景噪声被大量识别为语音

- 提高 threshold。
- 增大最短语音长度。
- 增加目标噪声域验证数据，而不是只凭听感调参。

### 自定义模型训练 loss 很低但真实效果差

教学数据的全段 0/1 标签过于简单，模型可能只学到能量或录音环境。需要真实帧级标注、语音噪声混合和独立录音级验证集。

## 14. 面试表达模板

> 我使用 Silero VAD 作为 ASR 前置切分。输入统一为单声道 16kHz，以 512 样本、32ms 窗口流式推理，并为每个会话独立维护循环状态。后处理没有直接使用单阈值，而是使用进入和退出不同的迟滞阈值，再结合最短语音和静音 hangover 合并时间段。我在 M5 上比较了 JITScript 与 ONNX Runtime，并扫描线程数。最终阈值不只看帧级 F1，还联合观察切分后的 Whisper WER、延迟和静音算力节省。

常见追问：

- 为什么 VAD 更关注 recall：漏掉的语音无法由后续 ASR 恢复。
- 为什么要迟滞阈值：避免概率在边界附近抖动导致频繁开关。
- 为什么每个会话需要独立 state：模型是有状态的，跨用户复用会污染上下文。
- 为什么不盲目量化：模型已经很小，真实瓶颈可能是 IO、线程调度和后续 ASR。
- 自定义 TinyVAD 的局限：当前教学标签是整段 0/1，不是生产级帧标注。

## 15. 建议练习顺序

1. 用官方模型切分 JFK 示例。
2. 分别测试 threshold 0.35、0.5、0.65。
3. 调整最短静音，观察切段数量变化。
4. 用合成数据训练 TinyVAD，确认形状和 loss。
5. 准备少量真实 speech/noise 数据重新训练。
6. 扫描 1、2、4 个线程，比较 JIT 和 ONNX。
7. 把切分结果送入 Whisper，联合记录 WER 和延迟。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
