# InsightFace 人脸识别实战教程：训练、参数、推理与调优

本教程介绍人脸检测、五点对齐、ArcFace embedding、验证阈值和 ONNX 静态量化，默认环境为 MacBook Air M5、24GB 统一内存。

对应代码：

- [`insightface_arcface/train.py`](../../insightface_arcface/train.py)：轻量 ArcFace 学生模型训练
- [`insightface_arcface/infer.py`](../../insightface_arcface/infer.py)：SCRFD + ArcFace 官方模型推理
- [`insightface_arcface/optimize.py`](../../insightface_arcface/optimize.py)：SCRFD/ArcFace ONNX 静态 PTQ
- [`insightface_arcface/README.md`](../../insightface_arcface/README.md)：原理和面试知识点

## 1. 学习目标

完成本教程后，应当能够：

1. 解释检测、关键点、对齐、embedding 和比对的完整链路。
2. 使用 SCRFD 和 ArcFace 验证两张照片是否为同一人。
3. 理解相似变换为什么对识别精度重要。
4. 使用 ArcFace 角度间隔损失训练轻量学生模型。
5. 调整 margin、scale、阈值、冻结策略和 AMP。
6. 使用 ONNX Runtime QDQ PTQ 优化检测和识别模型。

## 2. 先区分两条模型路径

| 路径 | 模型 | 用途 |
|---|---|---|
| 官方推理 | buffalo_l：SCRFD + `w600k_r50` ArcFace | 高质量人脸验证 |
| 教学训练 | MobileNetV2 + 128 维 embedding + ArcFace loss | 学习训练原理和轻量适配 |

[`infer.py`](../../insightface_arcface/infer.py) 使用官方 ONNX 模型；[`train.py`](../../insightface_arcface/train.py) 训练的是独立的轻量学生模型。训练出的 `arcface_student.pt` 不会被官方推理脚本自动加载。

要部署学生模型，需要另外实现学生模型的 checkpoint 加载、与训练一致的预处理、embedding 归一化和阈值标定。

## 3. 工业人脸识别链路

```text
原始图片
  → SCRFD 检测人脸框和 5 个关键点
  → 选择目标人脸
  → 5 点相似变换对齐到 112×112
  → ArcFace 网络输出 embedding
  → L2 normalize
  → cosine similarity
  → threshold
  → same / different
```

人脸识别不是“把整张照片直接送给分类器”。检测和对齐错误会直接污染 embedding，常常比最后的余弦公式更影响系统效果。

## 4. SCRFD 人脸检测

推理脚本把图片 letterbox 到 640×640，并在 stride 8、16、32 三个尺度解码输出：

- 人脸置信度。
- 边界框到 anchor center 四边的距离。
- 左眼、右眼、鼻尖、左嘴角、右嘴角五个关键点。

不同尺度负责大小不同的人脸。解码后执行 NMS，得到最终检测结果。

### 4.1 `det_10g` 的含义

模型名中的 `10g` 表示大约 10 GFLOPs 的计算规模，不表示 10GB，也不是参数数量。

## 5. 为什么必须做人脸对齐

ArcFace 训练时看到的是按照标准五点位置对齐的人脸。推理脚本使用标准 112×112 模板：

```text
左眼  右眼
   鼻尖
左嘴角 右嘴角
```

根据检测关键点和目标模板估计 similarity transform，包括旋转、统一缩放和平移：

\[
x'=sRx+t
\]

相似变换不会引入任意剪切，能保留人脸几何结构。仅按检测框 resize 会让不同姿态、旋转和框偏移造成 embedding 不稳定。

## 6. 第一次推理

```bash
cd /Users/yu/Code/ml-models-cookbook
uv run python -m insightface_arcface.infer --threads 4
```

第一次运行会下载约 280MB 的 `buffalo_l`。脚本只加载其中必需的：

- `det_10g.onnx`：SCRFD 检测。
- `w600k_r50.onnx`：ArcFace 识别。

默认两张示例图是不同的人，因此正常情况下相似度应低于阈值。

比较自己的照片：

```bash
uv run python -m insightface_arcface.infer \
  --a /绝对路径/photo_a.jpg \
  --b /绝对路径/photo_b.jpg \
  --thresh 0.35 \
  --det-thresh 0.5 \
  --threads 4
```

## 7. 推理参数怎么调

### 7.1 `--thresh`

余弦相似度阈值，默认 `0.35` 仅作为演示起点：

- 提高阈值：降低误接受，增加误拒绝。
- 降低阈值：减少误拒绝，增加误接受。

正式系统必须按目标数据和安全等级标定阈值。门禁、相册聚类和社交推荐不能共享同一个阈值。

### 7.2 `--det-thresh`

人脸检测置信度阈值，默认 `0.5`：

- 提高：减少背景误检，但可能漏掉小脸、侧脸或模糊脸。
- 降低：提高召回，但会让后续 ArcFace 处理更多错误区域。

检测 threshold 和识别 threshold 是两个不同概念。

### 7.3 `--threads`

控制 ONNX Runtime intra-op 线程数。建议测试 1、2、4 和默认值 0：

```bash
uv run python -m insightface_arcface.infer --threads 1
uv run python -m insightface_arcface.infer --threads 2
uv run python -m insightface_arcface.infer --threads 4
```

更多线程不保证更快。桌面应用还要考虑线程对 UI、视频解码和其他模型的影响。

## 8. ArcFace embedding 和相似度

官方识别模型先输出向量，再做 L2 归一化：

\[
\hat e=\frac{e}{\|e\|_2}
\]

两个归一化向量的点积就是余弦相似度：

\[
score=\hat e_1^T\hat e_2
\]

归一化消除了向量模长差异，使训练和推理集中在角度空间。

## 9. 准备学生模型训练数据

目录结构：

```text
data/faces/
├── person_001/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── person_002/
│   ├── 001.jpg
│   └── ...
└── ...
```

训练脚本使用 `ImageFolder`，目录名就是身份类别。

重要要求：训练图片应先经过人脸检测和五点对齐。虽然脚本会执行 `RandomResizedCrop(112)`，但随机裁剪不能替代几何对齐。

数据划分应按原始视频、拍摄会话或设备隔离，避免同一连拍中的近重复图片同时进入训练和验证。

## 10. 第一次训练：合成数据冒烟实验

如果数据目录不存在，脚本会生成 64 张随机图片和 8 个随机类别：

```bash
uv run python -m insightface_arcface.train \
  --data data/not-exist \
  --epochs 1 \
  --batch 16 \
  --freeze-backbone \
  --device mps \
  --out runs/arcface-smoke.pt
```

随机数据只用于验证 MobileNetV2、ArcFace margin、MPS 反向传播和 checkpoint 保存。训练准确率没有业务意义。

## 11. 第一次真实训练

小数据先冻结骨干：

```bash
uv run python -m insightface_arcface.train \
  --data data/faces \
  --epochs 20 \
  --batch 32 \
  --lr 1e-4 \
  --margin 0.5 \
  --scale 64 \
  --freeze-backbone \
  --device mps \
  --amp \
  --seed 42 \
  --out runs/arcface-head.pt
```

全量微调对照：

```bash
uv run python -m insightface_arcface.train \
  --data data/faces \
  --epochs 20 \
  --batch 16 \
  --lr 3e-5 \
  --margin 0.4 \
  --scale 64 \
  --device mps \
  --amp \
  --seed 42 \
  --out runs/arcface-full.pt
```

两条命令都是从 ImageNet MobileNetV2 权重开始的独立实验。当前脚本不会自动把冻结阶段的 checkpoint 接到全量微调阶段。严格的两阶段训练需要显式加载第一阶段 `model` 和 `arcface` state dict。

## 12. 训练模型结构

教学模型使用：

```text
112×112 RGB
  → MobileNetV2 features
  → global average pooling
  → BatchNorm1d(1280)
  → Dropout(0.3)
  → Linear
  → 128 维 embedding
  → ArcMarginProduct
  → CrossEntropyLoss
```

训练 checkpoint 保存：

- embedding 网络权重 `model`。
- ArcFace 分类权重 `arcface`。
- 身份类别列表 `classes`。
- 所有训练配置 `config`。

推理验证时通常只需要 embedding 网络，不需要训练身份分类权重；新身份通过注册 embedding 加入，不必重新训练整个分类器。

## 13. ArcFace loss 原理

对 embedding 和类别权重分别 L2 归一化后，普通分类 logit 是夹角余弦：

\[
cos\theta_j=\hat e^T\hat W_j
\]

ArcFace 对目标类别增加角度 margin：

\[
logit_y=s\cdot cos(\theta_y+m)
\]

非目标类别保持：

\[
logit_j=s\cdot cos\theta_j
\]

模型必须让目标类别角度更小，才能抵消额外的 `m`，从而得到更紧的类内分布和更大的类间边界。

代码通过三角恒等式计算目标 logit：

\[
cos(\theta+m)=cos\theta cos m-sin\theta sin m
\]

并在非单调区间进行修正，避免困难样本产生错误方向的梯度。

## 14. 训练参数怎么调

### 14.1 `--margin`

默认 `0.5`：

- margin 大：边界严格，潜在区分能力强，但更难收敛。
- margin 小：训练容易，但 embedding 的类间分离要求较弱。
- 小数据或噪声标签可从 0.3、0.4、0.5 扫描。

身份标签错误对 margin loss 的伤害很大，因为模型会强迫错误样本靠近错误中心。

### 14.2 `--scale`

归一化余弦值范围太小，直接 Softmax 会导致梯度不足。scale 默认 64，用来放大 logits。

常见搜索范围为 30～64。scale 和 margin 应联合调参。

### 14.3 `--freeze-backbone`

冻结 MobileNetV2 features，只训练后端 BN、embedding 层和 ArcFace 权重。代码还把冻结的 features 切换到 eval 模式，避免其 BatchNorm 统计量继续变化。

目标域与 ImageNet 差异较大时，完全冻结可能欠拟合，可以先冻结稳定训练，再逐步解冻。

### 14.4 `--lr`

- 冻结骨干：`1e-4`～`3e-4`。
- 全量微调：`1e-5`～`1e-4`。
- margin 增大后若 loss 不稳定，应适当降低学习率。

脚本使用 AdamW 和 cosine schedule。

### 14.5 `--amp` / `--no-amp`

MPS 上默认对 MobileNetV2 前向启用 FP16 autocast。ArcFace margin 和交叉熵强制回 FP32，因为平方根、角度边界和大 scale 对半精度更敏感。

出现 NaN 时先尝试：

```bash
--no-amp
```

再检查学习率、输入数值和 margin。

### 14.6 `--batch`

batch 过小时，每批身份和样本多样性不足；batch 过大则增加内存并降低梯度噪声。M5 24GB 可从 32 开始，全量微调内存不足时降到 16 或 8。

正式人脸训练通常使用 identity-balanced sampler，使每批包含多个身份和每个身份的多张图。当前教学脚本使用普通随机采样。

## 15. 人脸验证评估

不能只看训练分类准确率。验证模型需要构造：

- 正样本 pairs：同一身份的不同图片。
- 负样本 pairs：不同身份的图片。

然后统计：

- ROC 曲线。
- EER。
- TAR@FAR，例如 `TAR@FAR=1e-4`。
- 不同年龄、姿态、光照、设备等切片指标。

高安全系统还必须评估照片、屏幕回放、面具和生成式攻击。ArcFace 本身不是活体检测模型。

## 16. 性能优化实验

先只量化 ArcFace，加快实验：

```bash
uv run python -m insightface_arcface.optimize \
  --runs 50 \
  --threads 4 \
  --rebuild \
  --skip-detector
```

再处理完整链路：

```bash
uv run python -m insightface_arcface.optimize \
  --runs 50 \
  --threads 4 \
  --rebuild
```

脚本对官方 ONNX 模型执行 QDQ 静态 PTQ：

```text
FP32 ONNX
  → 真实人脸/图片校准激活范围
  → QDQ int8 ONNX
  → 延迟、大小和输出一致性比较
```

由于 `buffalo_l` 模型使用较老的 ONNX opset，当前实现使用 per-tensor 量化，避免 per-channel QDQ 的 axis 兼容问题。

## 17. 如何验证量化结果

ArcFace 检查：

- FP32 与 int8 延迟。
- 模型文件大小。
- 同一输入的 embedding cosine。

SCRFD 检查：

- FP32 与 int8 延迟。
- 模型文件大小。
- 同一图片检测到的人脸数量。

这些只是 smoke checks。当前脚本固定使用两张示例图片进行校准，不足以支持生产精度结论。正式量化需要扩展为代表性校准集，并重新评估：

- 检测 AP、召回率和关键点误差。
- 人脸验证 ROC、EER、TAR@FAR。
- 整条检测—对齐—识别链路的最终错误率。

`--rebuild` 用于在量化策略或校准输入改变后重建 int8 文件。

## 18. 端到端性能拆分

人脸验证总延迟可以拆成：

```text
图片解码
+ SCRFD
+ NMS
+ 五点对齐
+ ArcFace
+ embedding 比对
```

如果图片中有很多人脸，ArcFace 会执行多次。优化时应分别测量单脸和多人场景，而不是只测一个孤立的识别网络。

可进一步优化：

- 视频中使用检测间隔和人脸跟踪。
- 只对新轨迹或质量足够的人脸计算 embedding。
- 对已注册 embedding 做批量矩阵乘法。
- 缓存稳定轨迹的 embedding，并对多帧 embedding 聚合。
- 低质量、小尺寸或大姿态人脸先拒绝，避免不可靠比对。

## 19. 常见故障

### 检测不到人脸

- 降低 `--det-thresh`。
- 检查图片是否能被 OpenCV 读取。
- 人脸可能太小、太模糊或侧脸角度过大。
- 检查颜色空间和 letterbox 是否被修改。

### 同一个人相似度很低

- 关键点或对齐错误。
- 图片过度模糊、遮挡或姿态差异太大。
- 人脸裁剪尺寸过小。
- 阈值直接照搬其他数据集。

### 不同人相似度过高

- 阈值未按目标域标定。
- 误选了图片中的另一张脸。
- 注册图数量太少或质量差。
- 测试 pairs 或身份标签有错误。

### 量化后框或 embedding 异常

- 重建 int8 文件，避免复用旧缓存。
- 增加校准覆盖度。
- 对敏感层保留 FP32。
- 分别定位检测、对齐和识别哪个阶段发生偏移。

## 20. 隐私和安全边界

人脸 embedding 属于敏感生物特征。实际系统需要考虑：

- 明确授权和数据最小化。
- 静态和传输加密。
- 模板撤销或版本更新策略。
- 访问审计和保留期限。
- 活体检测与重放攻击。
- 不同人群上的公平性和误差切片。

模型精度高不代表系统自动满足合规和安全要求。

## 21. 面试表达模板

> 我实现了 SCRFD 检测、五点相似变换对齐、ArcFace embedding 和余弦阈值判定的完整人脸验证链路。对齐使用旋转、统一缩放和平移，将五个关键点映射到 112×112 标准模板，避免仅按框缩放造成姿态偏差。训练实验使用 MobileNetV2 学生骨干和 ArcFace angular margin loss，并在 MPS AMP 下把 margin 计算保留为 FP32。部署优化采用 ONNX Runtime QDQ 静态 PTQ，分别验证检测数量、embedding 一致性、延迟和模型大小。最终阈值和量化精度使用目标域 ROC、EER 与 TAR@FAR 标定，而不是依赖单张示例。

常见追问：

- 为什么对齐重要：识别模型训练分布就是标准对齐人脸。
- ArcFace 比普通 Softmax 多了什么：在归一化角度空间为目标类加入固定 margin。
- 为什么要 scale：归一化余弦范围太小，需要放大以获得合适梯度。
- 为什么单个 embedding cosine 不足以验证量化：它不代表完整 pair 分布和阈值处的错误率。
- 人脸识别和活体检测是否相同：不同；ArcFace 只做身份表征。

## 22. 建议练习顺序

1. 运行默认两张示例图，观察框、置信度和相似度。
2. 用同一人的两张照片测试，并改变识别阈值。
3. 改变检测阈值，观察小脸和误检。
4. 用合成数据跑通学生模型训练。
5. 准备已经对齐的多身份 ImageFolder。
6. 对比冻结骨干和全量微调。
7. 构造验证 pairs 并计算 ROC/EER。
8. 分别量化 ArcFace 和完整链路。
9. 使用更完整的校准集重新评估检测与验证精度。

实验结果可记录到 [`docs/EXPERIMENT_TEMPLATE.md`](../EXPERIMENT_TEMPLATE.md)。
