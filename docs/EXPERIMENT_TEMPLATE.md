# 实验记录模板

复制本页为一次实验记录。目标是让另一位工程师能够复现结论，而不是只留下“int8 快了 2 倍”。

## 实验问题

- 假设：
- 业务约束：延迟 / 吞吐 / 内存 / 精度中哪一个优先？
- 对照组：
- 唯一主动变量：

## 环境

| 项 | 值 |
|---|---|
| 日期 | |
| Git commit | |
| 机器 | MacBook Air M5, 24GB |
| macOS | |
| Python | |
| PyTorch / torchvision | |
| ONNX Runtime | |
| 后端 | CPU / MPS / CoreML |
| CPU 线程数 | |
| 电源模式 | |

## 数据与模型

| 项 | 值 |
|---|---|
| 模型/权重 | |
| 数据版本与切分 | |
| 输入 shape/时长 | |
| batch | |
| 预处理 | |
| 校准集 | 不适用 / 数量与来源 |

## 训练参数

| 参数 | 值 |
|---|---|
| seed | |
| optimizer | |
| learning rate / schedule | |
| weight decay | |
| epochs | |
| freeze 策略 | |
| AMP | |
| gradient accumulation | |

## 结果

| 方案 | 任务指标 | mean latency | p50 | p95 | throughput | 模型大小 | 峰值内存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | | | | | | | |
| optimized | | | | | | | |

## 正确性检查

- 输出一致性：
- 量化前后任务指标变化：
- 阈值是否重新标定：
- 失败样本/回归样本：

## 结论

- 是否接受该优化：
- 收益来自哪里：计算、访存、框架开销还是输入变化？
- 代价和适用边界：
- 下一步实验：
