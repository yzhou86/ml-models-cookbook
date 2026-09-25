# M5 本机验证记录

本页记录 2026-09-25 的 smoke test，目的是证明关键路径可运行并暴露优化的真实方向。它不是正式精度报告，也不是不同机器之间的性能承诺。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Apple M5, arm64, 24GB 统一内存 |
| macOS | 27.0 |
| Python | 3.12.13 |
| PyTorch / torchvision / torchaudio | 2.14.0 / 0.29.0 / 2.11.0 |
| Transformers | 4.57.6 |
| ONNX Runtime | 1.30.0 |
| MPS | available=True |
| PyTorch quantized engine | qnnpack |

依赖来自仓库中的 `uv.lock`。代码规范和无下载回归测试结果：

```text
ruff check .  -> passed
pytest        -> 6 passed
```

## 功能验证

| 模块 | 验证内容 | 结果 |
|---|---|---|
| Whisper | tiny、11 秒 JFK 音频、MPS 英文转写 | 成功，完整输出目标句 |
| Whisper 微调 | 2 train / 1 eval、冻结 Encoder、MPS | 完成前向、反向、生成评估和保存 |
| Silero | 官方 JIT、512 点滑窗、时间戳合并 | 检出 5 个语音段并保存切片 |
| 自定义 VAD | 合成数据、8 段、1 epoch、MPS | shape 匹配，完成训练与 checkpoint |
| ECAPA | SpeechBrain 1.1、同音频验证 | 192 维 embedding，cosine=1.0 |
| ECAPA 适配 | 两个临时说话人、embedding 缓存、MPS 头 | 完成 1 epoch 与保存 |
| MobileNetV3 | 合成数据、冻结骨干、MPS AMP | 完成 1 epoch 与保存 |
| YOLO11n | coco8、64 输入、batch=2、MPS | 完成 1 epoch、验证和 best.pt |
| InsightFace | Lena 与 t1、SCRFD→对齐→ArcFace | 检测成功，两个不同人 cosine=0.0358 |
| ArcFace 学生 | 合成数据、冻结骨干、MPS AMP | 完成 1 epoch 与保存 |

## Smoke benchmark

以下数字来自很少的正式计时次数，适合检查优化路线，不适合作为简历最终数据。正式报告请按实验模板增加预热、重复次数和真实验证集。

### Whisper tiny

设置：生成 5 个 token，`runs=1`。

| 路线 | 延迟 | state_dict 大小 | 输出 |
|---|---:|---:|---|
| CPU fp32 | 125.67ms | 151.1MB | `And so my fellow Americans` |
| CPU dynamic int8 | 218.52ms | 121.8MB | 相同 |
| MPS fp32 | 64.11ms | 同 fp32 | 相同 |

结论：该输入上动态 int8 虽然缩小权重，却更慢。原因可能是短解码、小矩阵和动态量化开销；这正说明不能把“int8”直接等同于“更快”。

### Silero VAD

设置：11 秒音频，包含完整滑窗循环，CPU threads=2，`runs=1`。

| 路线 | 端到端窗口循环延迟 |
|---|---:|
| JIT | 49.16ms |
| ONNX Runtime | 21.84ms |

### ECAPA-TDNN 骨干

设置：同一组归一化 fbank，`runs=1`。

| 路线 | 延迟 | 大小 |
|---|---:|---:|
| fp32 | 49.23ms | 83.3MB |
| dynamic int8 Linear | 47.97ms | 83.3MB |

结论：模型由 Conv1d 主导，只量化 Linear 几乎没有大小收益。面试中应明确说明“该优化不值得采用”。

### MobileNetV3-Large

设置：batch=1、224×224、单张示例图校准、`runs=1`。

| 路线 | 延迟 | 大小 |
|---|---:|---:|
| CPU fp32 | 43.44ms | 22.1MB |
| CPU dynamic | 44.43ms | 14.6MB |
| CPU static PTQ | 3.98ms | 5.6MB |
| MPS fp32 | 6.60ms | 22.1MB |

单图校准的 logits cosine 只有 0.6316，虽然该图 top-1 都是 Samoyed，但不能据此接受量化模型。正式 PTQ 必须使用代表性校准集并测 ImageNet/业务验证集。

### YOLO11n

设置：320 输入、相同 bus 图片、完整预处理+模型+NMS、`runs=1`。

| 路线 | 端到端延迟 | 模型大小 |
|---|---:|---:|
| PyTorch CPU | 10.67ms | 5.4MB `.pt` |
| PyTorch MPS | 10.42ms | 同上 |
| ONNX CPU | 8.18ms | 10.6MB |
| ONNX QDQ int8 CPU | 7.30ms | 4.5MB |

量化时保留 DFL、sigmoid 和坐标解码所在检测头为 fp32；全图量化曾产生大于 1 的异常置信度，说明敏感后处理节点不应盲目量化。

### InsightFace

设置：两张校准图片、CPU threads=4、`runs=1`。

| 模型 | fp32 延迟 | int8 延迟 | fp32 大小 | int8 大小 | 一致性 |
|---|---:|---:|---:|---:|---|
| ArcFace R50 | 26.65ms | 14.98ms | 174.4MB | 43.8MB | embedding cosine=0.9790 |
| SCRFD det_10g | 52.69ms | 29.35ms | 16.9MB | 4.3MB | 两者均检出 1 张脸 |

两张图片只够验证量化图可以加载和执行。正式验收必须扩大校准集，并报告 ROC、EER、TAR@FAR、检测召回和关键点误差。

## 尚未纳入本次验证

- Whisper base/small 的完整训练；它们有安全配置，但需要更长运行时间。
- CoreML 导出与 Xcode/ANE 性能；`coremltools` 是可选依赖，需 `uv sync --extra apple`。
- 各任务真实数据集上的最终精度与长时间热稳定性。
- `torch.compile` 的稳态收益；MPS 和自回归模型可能出现图断裂，应作为独立实验。
