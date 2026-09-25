# ML Models Cookbook

面向算法工程面试和 Apple Silicon 实践的六模型实验集，覆盖训练、推理、量化、ONNX、CoreML、MPS 与性能验证。默认目标机器是 **MacBook Air M5 / 24GB 统一内存**；所有默认配置都优先保证这类无风扇设备能够完成，而不是追求服务器级吞吐。

## 先读这三份文档

- [M5 安装、调参与性能测试](docs/M5_SETUP_AND_BENCHMARK.md)
- [面试知识体系与高频追问](docs/INTERVIEW_GUIDE.md)
- [实验记录模板](docs/EXPERIMENT_TEMPLATE.md)
- [本机验证矩阵与 smoke benchmark](docs/VALIDATION_M5.md)

每个模型目录还有独立 README，包含架构、损失、指标和业务落地问题。

## 模型地图

| 模型 | 任务 | 默认运行后端 | 本工程重点 |
|---|---|---|---|
| Whisper tiny/base/small | ASR | MPS 浮点 / CPU int8 | 冻结 Encoder、梯度累积、动态量化、compile |
| Silero VAD | 语音切段 | CPU | 有状态滑窗、迟滞阈值、JIT vs ONNX |
| ECAPA-TDNN | 声纹 | CPU 骨干 / MPS 分类头 | embedding 缓存、AAM-Softmax、动态量化 |
| YOLO11n | 检测 | MPS / ONNX CPU / CoreML | 端到端公平基准、静态 PTQ、导出 |
| MobileNetV3 | 分类 | MPS / CPU int8 | 迁移学习、冻结 BN、静态 PTQ |
| SCRFD + ArcFace | 人脸验证 | ONNX CPU | 检测、5 点相似变换、QDQ 静态 PTQ |

> 重要原则：MPS 适合浮点训练与推理；PyTorch int8/qnnpack 和 ONNX Runtime int8 主要运行在 CPU。小 batch、小模型不保证 GPU 更快，必须实测。

## 在 M5 上安装

推荐让 `uv` 管理独立 Python 3.12，避免系统 Python 或过新的 Python 缺少 ML wheel：

```bash
brew install uv
uv python install 3.12
uv sync --python 3.12

# 需要 YOLO CoreML 导出时
uv sync --python 3.12 --extra apple
```

先跑不下载大模型的回归测试：

```bash
uv run pytest
uv run ruff check .
```

首次运行各模块会下载公开预训练权重。InsightFace `buffalo_l` 约 280MB，Whisper/ECAPA 也会写入各自框架的用户缓存。

## 推理

```bash
uv run python -m whisper_asr.infer --model tiny --language en
uv run python -m silero_vad.infer
uv run python -m ecapa_tdnn.infer
uv run python -m mobilenet_v3.infer --variant small
uv run python -m yolo11n.infer --device auto
uv run python -m insightface_arcface.infer --threads 4
```

所有音频入口都会转为单声道 16kHz。`--device auto` 在 Apple Silicon 上选择 MPS；ECAPA 默认 CPU，因为 SpeechBrain 的部分算子在 MPS 上覆盖不完整。

## M5 24GB 安全训练预设

```bash
# 先冻结 Whisper Encoder；等效 batch=4
uv run python -m whisper_asr.train_finetune \
  --model tiny --batch 2 --grad-accum 2 --epochs 2

# 无真实数据时生成合成数据验证形状和训练循环
uv run python -m silero_vad.train_custom_vad --epochs 2 --samples-per-class 32

# ImageFolder 迁移学习，先冻结骨干、160 分辨率
uv run python -m mobilenet_v3.train \
  --data data/images --freeze --imgsz 160 --batch 32 --epochs 10

# Ultralytics 自动下载 coco8；320 用于快速调参
uv run python -m yolo11n.train --imgsz 320 --batch 16 --epochs 10

# 已对齐的人脸 ImageFolder；小数据先冻结 MobileNetV2 骨干
uv run python -m insightface_arcface.train \
  --data data/faces --freeze-backbone --batch 32 --epochs 10

# 先预计算冻结的 ECAPA embedding，再快速训练业务分类头
uv run python -m ecapa_tdnn.train --data data/spk --batch 64 --epochs 20
```

如果出现内存压力，依次降低 `batch`、输入分辨率或模型尺寸，再开启梯度检查点；不要把 swap 当成正常训练内存。

## 优化与可复现实验

```bash
uv run python -m whisper_asr.optimize --model tiny --runs 5 --skip-compile
uv run python -m silero_vad.optimize --threads 4 --runs 20
uv run python -m ecapa_tdnn.optimize --runs 20

# 正式 PTQ 应传 100~500 张部署分布图片
uv run python -m mobilenet_v3.optimize --calib-dir data/calibration --runs 20
uv run python -m yolo11n.optimize --calib-dir data/calibration --runs 10 --coreml
uv run python -m insightface_arcface.optimize --threads 4 --runs 20
```

优化脚本遵循四条规则：

1. 先预热，再同步 MPS/CUDA 后计时。
2. 比较端到端延迟时使用相同输入、预处理和后处理。
3. 除延迟外，同时报告模型大小和输出一致性。
4. 示例数据只能证明流程可运行；量化精度必须在真实验证集上报告。

建议至少记录 latency mean/p50/p95、吞吐、峰值内存、模型大小和任务指标。不要只展示“快了几倍”。

## 可调参数入口

| 类别 | 常用参数 | 主要权衡 |
|---|---|---|
| 训练规模 | `--batch`, `--grad-accum`, `--imgsz` | 内存、吞吐、收敛稳定性 |
| 优化器 | `--lr`, `--weight-decay`, `--warmup-ratio` | 收敛速度与过拟合 |
| 迁移学习 | `--freeze`, `--freeze-backbone`, `--train_encoder` | 训练成本与域适配能力 |
| 度量学习 | `--margin`, `--scale` | 类内紧致、类间间隔与训练难度 |
| 部署 | `--threads`, `--runs`, `--calib-dir` | CPU 并发、测量方差、int8 精度 |
| 检测/VAD | `--conf`, `--threshold`, `--min-silence-ms` | precision、recall 与响应延迟 |

## 目录结构

```text
common/                   公共设备、下载、音频、随机种子和 benchmark
whisper_asr/              ASR
silero_vad/               官方 VAD 推理 + 自定义轻量 VAD
ecapa_tdnn/               说话人 embedding / 分类适配
mobilenet_v3/             图像分类与 PTQ
yolo11n/                  目标检测、ONNX、CoreML
insightface_arcface/      人脸检测、对齐、识别与 PTQ
tests/                    无需下载模型的快速回归测试
docs/                     M5 实操、面试与实验模板
```

运行产物默认写入 `samples/`、`runs/` 或命令指定的输出位置，并已加入 `.gitignore`。
