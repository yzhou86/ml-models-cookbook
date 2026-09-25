# ML Models Cookbook（uv + Apple Silicon M5 面试实战工程）

面向算法岗面试的 **小模型训练 / 推理 / 优化加速** 实战工程，覆盖 **语音** 与 **视觉** 两大赛道共 6 个模型。所有代码在 **MacBook Air M5（24GB）** 上验证可行：训练走 `mps` 后端，量化推理走 CPU（`qnnpack` 后端）。依赖由 [uv](https://docs.astral.sh/uv/) 管理。

## 六大模型速览

| # | 模型 | 任务 | 参数量 | 核心架构 | 本工程演示的优化 |
|---|---|---|---|---|---|
| 1 | **Whisper tiny/base/small** | 语音识别 ASR | 39M / 74M / 244M | Encoder-Decoder Transformer（log-mel → 文本） | int8 动态量化、torch.compile、冻结 encoder 微调 |
| 2 | **Silero VAD** | 语音活动检测 | ~2M | 1D-CNN + GRU（512 样本滑窗） | ONNX Runtime 加速、自训轻量 VAD |
| 3 | **ECAPA-TDNN** | 说话人识别/声纹 | ~22M | Res2Net 风格 TDNN + SE-Block + AAM-Softmax | int8 量化、TorchScript、冻结骨干微调 |
| 4 | **YOLO11n** | 目标检测 | ~2.6M | CSPNet backbone + PAFPN 颈部 + 解耦头 | ONNX 导出 + int8 量化 + CoreML 原生部署 |
| 5 | **MobileNetV3** | 图像分类 | 2.5M(small) / 5.4M(large) | 深度可分离卷积 + SE + h-swish | 静态 PTQ（校准量化）、动态量化、AMP 训练 |
| 6 | **InsightFace (SCRFD + ArcFace)** | 人脸检测+识别 | 16M + 88M(ONNX) | SCRFD anchor-free 检测 + 5 点对齐 + ArcFace 角度间隔 | ONNX int8 量化（170MB→45MB） |

## 项目结构

```
ml-models-cookbook/
├── pyproject.toml           # uv 依赖清单
├── common/utils.py          # 设备检测（mps/cuda/cpu）、计时、基准测试
├── whisper_asr/             # 1. Whisper（ASR）        + README.md 面试详解
├── silero_vad/              # 2. Silero VAD            + README.md 面试详解
├── ecapa_tdnn/              # 3. ECAPA-TDNN（声纹）     + README.md 面试详解
├── yolo11n/                 # 4. YOLO11n（检测）       + README.md 面试详解
├── mobilenet_v3/            # 5. MobileNetV3（分类）   + README.md 面试详解
├── insightface_arcface/     # 6. InsightFace（人脸）    + README.md 面试详解
└── samples/                 # 自动下载的示例数据
```

每个模块统一三件套：`train*.py`（训练）、`infer.py`（推理）、`optimize.py`（优化加速 + 基准对比），并配有一份可直接背的 `README.md` 面试详解。

## 安装（uv）

```bash
cd ml-models-cookbook

# 若尚未安装 uv（macOS）：
#   brew install uv
#   或官方脚本: curl -LsSf https://astral.sh/uv/install.sh | sh

# 一键创建虚拟环境并安装全部依赖（uv 自动解析 arm64 wheel）
uv sync

# 之后所有脚本用 uv run 启动，自动使用正确环境
uv run python -m whisper_asr.infer
```

单独添加依赖示例：`uv add onnxruntime`；进入环境：`source .venv/bin/activate`。

## 快速体验

```bash
uv run python -m whisper_asr.infer            # Whisper ASR（自动下载 whisper-tiny）
uv run python -m silero_vad.infer             # VAD 时间戳检测
uv run python -m ecapa_tdnn.infer             # 声纹验证（同段音频自比较）
uv run python -m yolo11n.infer                # YOLO 检测 + 可视化
uv run python -m mobilenet_v3.infer          # ImageNet 分类 Top5
uv run python -m insightface_arcface.infer   # 人脸比对（首次下载 ~280MB）

# 优化加速对比（面试演示重点）
uv run python -m mobilenet_v3.optimize       # 静态 PTQ 全流程 + 延迟对比
uv run python -m whisper_asr.optimize        # fp32 vs int8 vs compile
uv run python -m yolo11n.optimize             # ONNX + int8 + CoreML 导出
uv run python -m insightface_arcface.optimize # ArcFace ONNX 量化

# 训练
uv run python -m yolo11n.train                # YOLO（MPS）
uv run python -m mobilenet_v3.train           # 分类（AMP 半精度）
uv run python -m whisper_asr.train_finetune  # Whisper 冻结 encoder 微调
uv run python -m silero_vad.train_custom_vad  # 自训 100K 参数 VAD
uv run python -m insightface_arcface.train    # ArcFace loss 从零实现
uv run python -m ecapa_tdnn.train             # 声纹分类微调（需自备数据）
```

## 优化技术地图（面试总纲）

| 技术 | 原理一句话 | 本工程落点 |
|---|---|---|
| **动态量化** | Linear 权重 int8，激活动态量化，免校准 | whisper / ecapa / mobilenet / arcface |
| **静态量化 (PTQ)** | MinMax observer 校准 + Conv/BN/ReLU 融合 | `mobilenet_v3/optimize.py`（含 QuantStub 包装） |
| **ONNX Runtime** | 图优化 + 算子融合，跨平台推理引擎 | silero / yolo / arcface |
| **int8 ONNX 量化** | 权重 QInt8，零校准成本 | `yolo11n/optimize.py`、`insightface_arcface/optimize.py` |
| **torch.compile** | Inductor 算子融合 + kernel 生成 | whisper / mobilenet（失败自动降级） |
| **TorchScript** | 图模式执行，消除 Python 开销 | ecapa / silero |
| **CoreML** | macOS/iOS 原生 GPU 加速 | `yolo11n/optimize.py` 导出 |
| **AMP 混合精度** | 前向 fp16 反向 fp32，MPS 原生支持 | mobilenet / arcface 训练 |
| **冻结微调** | 只训头部/decoder，小内存适配 | whisper / ecapa |
| **知识蒸馏** | 教师(大)→学生(小) 软标签迁移 | 各模块 README 中讲解思路 |
| **AAM/ArcFace Loss** | 角度间隔拉大类间距离 | ecapa / insightface 训练实现 |

## Apple Silicon（M5 / 24GB）注意事项

- 设备优先级 `mps > cuda > cpu`，统一由 `common/utils.py:get_device()` 决定
- **量化模型只能跑 CPU**（qnnpack 后端），各 optimize 脚本已自动回退并保证对比公平
- `torch.compile` 对 MPS 后端覆盖有限，代码用 try/except 保护，失败自动降级 fp32
- SpeechBrain 的 ECAPA 部分算子 MPS 不支持，默认 CPU（22M 模型 CPU 推理 <50ms，够用）
- 不要安装 `onnxruntime-gpu`，Mac 用普通 `onnxruntime`（CPU EP）
- Whisper 训练显存不够时：冻结 encoder 或降到 tiny；M5 24GB 跑 small 全参微调可行但慢

## 面试冲刺路线建议

1. **先跑通**：6 个 `infer.py` 全部体验一遍，建立手感
2. **吃透两个深度案例**：`mobilenet_v3/optimize.py`（PTQ 完整原理）和 `insightface_arcface/infer.py`（检测→对齐→比对全链路）
3. **背熟各模块 README.md 的「面试 Q&A」小节**——按面试官视角的高频问题整理
4. **准备一条业务叙事线**：例如"长视频字幕生成：Silero VAD 切片 → Whisper int8 批量转写 → 字幕合并"，体现工程闭环能力
