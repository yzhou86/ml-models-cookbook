# MacBook Air M5（24GB）运行、调参与性能手册

## 1. 先理解这台机器的边界

Apple Silicon 使用统一内存：CPU、GPU 和系统共享 24GB。MPS 不需要把数据复制到独立显存，但模型权重、激活、优化器状态、数据缓存和桌面应用仍在争抢同一块内存。MacBook Air 无风扇，长时间训练还会因为温度产生降频。

实践中应关注四个量：

1. **模型权重**：fp32 每参数 4 字节；训练还需要梯度和优化器状态。
2. **激活**：通常随 batch、序列长度或图像面积线性增长，是训练内存大头。
3. **统一内存压力**：Activity Monitor 中出现大量 swap 时，训练速度会断崖式下降。
4. **持续功耗**：短 benchmark 和半小时训练的速度不可直接类比。

因此本项目默认采用小模型、较小 batch、冻结微调和短 benchmark。

## 2. 环境安装

### 推荐方案

```bash
brew install uv
cd ml-models-cookbook

uv python install 3.12
uv sync --python 3.12
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

最后一项应输出 `True`。如果为 `False`：

- 确认使用 arm64 Python，而不是 Rosetta 下的 x86 Python：`uname -m` 应为 `arm64`。
- 更新 macOS 和 Xcode Command Line Tools。
- 用 `uv run python ...`，不要误用系统 Python。
- 删除并重建 `.venv` 前先确认没有需要保留的本地产物。

CoreML 是可选依赖：

```bash
uv sync --python 3.12 --extra apple
```

### 为什么限定 Python 3.10–3.12

机器学习依赖包含大量原生 wheel。系统刚发布的新 Python 往往先于 PyTorch、ONNX Runtime、coremltools 的兼容周期。独立安装 Python 3.12 能减少“能解析依赖但没有 arm64 wheel”的问题。

## 3. 分层验证，不要一上来下载所有模型

```bash
# 第 1 层：语法、数学和张量形状，不下载权重
uv run pytest
uv run ruff check .

# 第 2 层：小权重推理
uv run python -m mobilenet_v3.infer
uv run python -m yolo11n.infer
uv run python -m silero_vad.infer

# 第 3 层：较大下载和完整流水线
uv run python -m whisper_asr.infer --model tiny
uv run python -m ecapa_tdnn.infer
uv run python -m insightface_arcface.infer
```

遇到错误时记录：命令、Python/PyTorch 版本、设备、输入尺寸和完整异常。只说“MPS 不工作”通常无法定位问题。

## 4. 训练配置建议

### Whisper

| 配置 | 建议起点 | 说明 |
|---|---:|---|
| 模型 | tiny | 先验证数据和 WER，再换 base/small |
| batch | 2 | small 建议从 1 开始 |
| grad accumulation | 2–8 | 增大等效 batch，不增加单步激活内存 |
| encoder | 冻结 | 小数据和第一次实验更稳 |
| gradient checkpointing | 内存不足再开 | 省激活，增加约一次重计算 |
| learning rate | 1e-5 | 解冻 Encoder 后通常还应更小 |

调参顺序：先固定 tiny 和数据切分，扫描学习率；再决定是否解冻 Encoder；最后才增加模型尺寸。否则模型、数据和优化器同时变化，无法解释收益来自哪里。

### MobileNetV3 / ArcFace 学生模型

- 先使用 `imgsz=160`、冻结骨干，确认标签和损失正常。
- 再解冻最后若干层或全骨干，用更小学习率训练。
- MPS AMP 可以节省激活，但不应预先承诺固定加速比例。
- 冻结骨干时必须让 BatchNorm 保持 eval；代码已处理，否则 running mean/variance 仍会漂移。
- 使用 ImageNet 预训练权重时必须使用对应 mean/std；代码已统一补齐。

### YOLO11n

- 调试：`imgsz=320, batch=16, coco8, epochs=10`。
- 正式小数据：`imgsz=640, batch=8` 起步，根据内存逐步加 batch。
- 小目标需要更大分辨率，但面积翻倍会使计算量约增加四倍。
- `workers=0` 最稳；数据量大且训练稳定后再试 2 或 4。
- 早期实验可冻结部分层，最终应在验证集确认解冻是否提升 mAP。

### ECAPA-TDNN

SpeechBrain 骨干默认放 CPU。冻结骨干时每条音频的 embedding 与 epoch 无关，因此代码会先计算一次，再只在小型张量上训练 AAM 头。这比每个 epoch 重复跑 ECAPA 更合理。

调 `margin` 时不要只看闭集分类准确率，应观察验证对上的 EER、ROC 和目标 FAR 下的 TAR。margin 越大并不必然越好；小数据上过大的 margin 会导致训练困难。

### 自定义 VAD

合成数据只能验证训练循环。真实数据至少应覆盖：

- 多种说话人、距离和响度；
- 稳态噪声与突发噪声；
- 纯静音、音乐、键盘、呼吸声；
- 语音与噪声混合，而不是只训练“纯语音 vs 纯噪声”。

当前教学数据把整段语音文件标为 1、噪声文件标为 0。生产训练应使用帧级时间标注或可靠的弱标注，并对边界区域做容忍处理。

## 5. 正确做性能实验

### 延迟和吞吐不是同一个指标

- latency：单请求完成时间，实时应用更关心 p50/p95/p99。
- throughput：单位时间处理的样本数，批处理服务更关心。
- realtime factor：处理时间 / 音频时长，ASR/VAD 常用；小于 1 才能实时。

batch 增大通常提升吞吐，却可能增加单请求延迟。面试中必须先说业务目标，再谈优化。

### 基准协议

1. 固定电源模式，关闭高负载后台程序。
2. 记录机型、macOS、Python、PyTorch、线程数和输入尺寸。
3. 至少预热 3 次；`torch.compile` 的编译时间单独记录。
4. MPS/CUDA 计时前后同步，否则只测到异步提交时间。
5. 比较相同边界：都测裸模型，或都测端到端，不要混合。
6. 至少重复 20 次，报告均值和分位数；长任务可减少次数。
7. 同时验证输出：分类 top-1、检测框、embedding cosine、WER/mAP/EER。

### CPU 线程

ONNX Runtime 和 PyTorch 的最优线程数与模型不同。M5 上建议实测 1、2、4、默认四组：

```bash
uv run python -m silero_vad.optimize --threads 1
uv run python -m silero_vad.optimize --threads 2
uv run python -m silero_vad.optimize --threads 4
uv run python -m silero_vad.optimize --threads 0
```

线程更多可能带来调度开销和温度压力，不一定更快。

## 6. 量化的正确验证方式

### 动态量化

运行时量化激活，通常只覆盖 Linear/MatMul。Transformer 的 Linear 比例高，因此比卷积网络更可能受益；MobileNet 动态量化只动分类头，主要价值是作为对照实验。

### 静态 PTQ

校准阶段用代表性数据观察激活范围，部署时权重和激活都可进入 int8。校准集不需要标签，但必须接近部署分布。建议：

- 100–500 个代表样本起步；
- 类别、亮度、尺寸、设备来源尽量均衡；
- 校准集和最终测试集分离；
- 量化后重新扫描业务阈值。

命令未传 `--calib-dir` 时，脚本只使用示例图做 smoke test，并会明确警告。该结果不能写进简历作为精度结论。

### 量化验收

| 任务 | 最低检查 | 正式检查 |
|---|---|---|
| 分类 | 同图 top-1、logits cosine | 验证集 top-1/top-5 |
| 检测 | 框数量和最高置信框 | mAP@0.5:0.95、各类别 AP |
| ASR | 解码文本是否一致 | WER/CER、实时率 |
| 声纹/人脸 | 同输入 embedding cosine | ROC、EER、TAR@FAR |
| VAD | 时间戳基本一致 | 帧 F1、段级召回、切段延迟 |

## 7. 常见故障

### MPS out of memory

先减 batch，再减图像尺寸/音频长度；Whisper 可冻结 Encoder 或开启 gradient checkpointing。退出占用统一内存的大应用后重试。不要把反复清空 MPS cache 当成根治方案。

### MPS 某算子不支持

这是后端覆盖问题，不一定是模型错误。先用 `--device cpu` 验证正确性；对单个不支持算子可考虑 CPU fallback，但频繁跨设备会很慢。ECAPA 因此默认全程 CPU 骨干。

### ONNX int8 没变快

检查量化图是否真的包含 Q/DQ 或整数算子、CPU provider 是否支持这些算子、模型是否被预处理/NMS 主导，以及线程设置是否合理。大小下降不等于延迟一定下降。

### torch.compile 首次运行特别慢

首次调用包含图捕获和编译，应单独统计。动态 shape、自回归生成和 MPS 算子覆盖都会导致图断裂；脚本允许 `--skip-compile`，失败也不会影响 eager 路线。

### 长时间训练越来越慢

MacBook Air 无风扇，可能热降频。记录每个 epoch 时间；正式比较可让机器冷却、固定电源状态，并避免把第一个和最后一个 epoch 混为同一热状态。
