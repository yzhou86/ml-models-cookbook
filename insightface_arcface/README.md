# InsightFace (SCRFD + ArcFace) 面试详解

## 1. 模型概况

| 项 | 说明 |
|---|---|
| 出处 | InsightFace 开源体系（邓健康等，2019 ArcFace 论文 + 2021 SCRFD 论文） |
| 任务 | 人脸检测（SCRFD）+ 人脸识别/验证（ArcFace） |
| buffalo_l 组件 | det_10g (SCRFD, 16MB) + w600k_r50 (ArcFace ResNet50, 170MB) |
| embedding | **512 维**，L2 归一化后余弦比对 |
| 训练集 | MS1MV3 (580 万人 510 万图，Glint360K 后继)、WIDER FACE（检测） |
| 精度 | LFW 99.83%+ / MegaFace 98% 时代基线；w600k_r50 是工业事实标准 |
| 1:1 vs 1:N | 验证（两张脸比对）vs 检索（在百万库中找，配 FAISS） |

本工程**不依赖 insightface pip 包**，用 onnxruntime 直接加载官方 ONNX 并手写解码/对齐——每一行都可以在面试白板上复现。

## 2. 全链路原理（面试标准答案：四步流水线）

```
原图 → ① SCRFD 人脸检测(bbox+5关键点)
      → ② 5 点相似变换对齐(112×112)
      → ③ ArcFace backbone → 512 维 embedding (L2 归一化)
      → ④ 余弦相似度 + 阈值判定
```

### 2.1 SCRFD 检测
- **Sample-aware/anchor-free**：在 8/16/32 三个 stride 特征图上直接回归 bbox（中心到四边距离×stride）与 5 关键点
- 训练损失：分类 Focal Loss（正负样本极不平衡）+ 回归 IoU loss
- 系列分档：det_500m(0.5MB)/det_10m/buffalo 系列——移动端 500m 即够，`g` = "gallon" 大杯版

### 2.2 5 点对齐（本工程 `align_face`）
用左/右眼角、鼻尖、左右嘴角五个点做**最小二乘相似变换**到标准模板（112×112 ArcFace 姿态坐标），消除姿态/尺度差异。比对算法对"正脸化"输入极度敏感，不做对齐精度大幅下降。

### 2.3 ArcFace 损失（本工程 `train.py` 完整实现，面试高频白板题）

普通 softmax：`W·x + b` 加性间隔。
**ArcFace 在角度空间加 margin**：

```
cos(θ_yi + m)          ← 正样本对数几率里加 m (m=0.5, s=64)
```

推导链（面试要求能写）：
1. 权重与特征 L2 归一化 → `logit = W_j·x = cosθ_j`
2. 正类 θ_yi → θ_yi + m，几何意义：类中心周围强制一条 angular margin 带
3. **效果**：类内压紧（θ→0）、类间推开（θ→π-m 以上）；embedding 空间呈"伞状簇"
4. s 缩放防止过小区间梯度消失

**变体对比**（必背）：
- CosFace：`cosθ - m`（余弦间隔，加在 logit 域）
- SphereFace：`cos(mθ)`（乘性，早期，训练不稳）
- ArcFace：`cos(θ+m)`（加性角度间隔，主流首选，收敛稳精度高）

### 2.4 比对与阈值
- L2 归一化后余弦 ∈ [-1,1]，实际业务阈值 **0.3~0.45**（官方 buffalo_l 约 0.35）
- 阈值按 FAR 预算定，不按"0.5 像不像"直觉定

## 3. 训练过程（本工程 `train.py`：轻量复现）

1. 数据：`data/faces/` 下每人一个目录若干图（LFW 结构），ImageFolder 加载
2. 骨干：MobileNetV2 迁移（128 维 embedding）——M5 可从零训练的教学配置
3. 头部：自实现 `ArcMarginProduct(s=64, m=0.5)`
4. 增强（人脸特有）：随机裁剪/翻转/**轻微旋转**，RGB/亮度抖动；**不可做过大几何形变**（对齐语义会被破坏）
5. 训练：AAM-Softmax CE + AdamW，20 epoch 内小数据即收敛

**工业级做法**（叙述给面试官）：
- 大规模 softmax 头（Glint360K 36 万类）用 partial FC / 权重采样策略缓解显存
- 噪声标签清洗（自动挖掘同人冲突样本）
- 检测与识别联合数据pipeline（WIDER FACE + 清洗后的识别集）

**评估协议**（必背）：
- LFW 1:1 验证（6000 对，accuracy）
- MegaFace 1:N 百万干扰库（rank-1），工业更贴近
- **拒绝牢记 FAR/FRR 曲线**：注册免验收紧 FAR、解锁场景看 FRR

## 4. 参数调整

| 参数 | 默认 | 经验 |
|---|---|---|
| margin m | 0.5 | 难例多（双胞胎）升 0.6~0.7；小数据/不收敛降 0.3 |
| scale s | 64 | 与类别数联动，一般 30~64 |
| embedding dim | 128~512 | 512 检索精度高；128 省存储（FAISS 库 4x 小） |
| 输入对齐 | 112×112 | 保持与骨干训练一致，随意改尺寸精度崩 |
| 阈值 | 0.35 (buffalo_l) | 按业务 FAR 定；跨模型不可复用阈值 |
| det 阈值 | 0.5 | 漏检多时降到 0.3 |

## 5. 训练速度优化

- **partial FC / 动态负类采样**：百万类 softmax 全权重算不动 → 每步采样一部分负类权重，显存/时间都降 90%+
- **AMP fp16**：骨干前向半精度
- **数据管道**：人脸图预对齐缓存（检测对齐一次，训练反复用）
- **冻结骨干训头部**：几百人小库注册场景（同 ECAPA 范式）

## 6. 推理优化（本工程 `optimize.py`）

| 手段 | 效果 |
|---|---|
| **ONNX int8 动态量化** | det 16MB→4MB、rec 170MB→**45MB**；CPU 提速 2x+；embedding 余弦 >0.99（脚本验证） |
| **换小骨干** | r50 → r18/r100 蒸馏版（MobileFaceNet 4MB）——移动端标准 |
| **CoreML EP** | macOS 原生 ANE 加速（onnxruntime-coreml） |
| **两级检索** | 检测全帧跑 → 只对检测框过识别网络（算力自动分配） |
| **1:N 检索** | embedding 入 **FAISS**（IVF-PQ 压缩索引）：百万库检索 <10ms；embedding 量化 PCA 512→128 |
| **批处理** | 多人脸视频帧拼 batch；静态图缓存 embedding |

**部署一句话架构**：SCRFD int8 → 对齐 → ArcFace int8 → FAISS 检索 → 阈值 + 活体检测（静默活体/红外/3D 结构光，识别层之外的独立风控层）。

## 7. 面试 Q&A

**Q1: 为什么 ArcFace 比 softmax 好？**
A: softmax 训练目标是可分，embedding 类内仍然松散、类间也没有明确间隔，1:N 检索时近邻干扰多。ArcFace 在角度域加 margin 强制类间几何间隔，embedding 度量结构直接可判别；且嵌入可用于未见类别（开集）。

**Q2: 人脸识别系统为什么必须先对齐？**
A: 识别骨干训练输入都是模板姿态，测试时侧脸/尺度差异会落入训练分布外；5 点对齐把输入拉回标准姿态，等价于极大降低类内方差。消融：不对齐 LFW 掉 3~5 个点。

**Q3: 1:N 佩戴眼镜/口罩/逆光误识别怎么办？**
A: ① 训练数据加 occlusion 增强 ② 检索层加 occlusion-aware 质量分支拒判低质量脸 ③ 业务层二次核验（1:1 复验 + 活体）④ 库内注册多姿态照片。

**Q4: 怎么防照片/屏幕攻击（活体检测）？**
A: 独立于识别的风控层：静默活体（单帧 CNN 判屏幕纹理/摩尔纹）、动作活体（眨眼摇头，时序模型）、红外双目、3D 结构光/ToF 深度。识别模型本身不解决活体，答"分层风控"加分。

**Q5: 百万人脸库 1:N 检索怎么做？**
A: embedding 离线入库 FAISS；用 IVF-PQ（倒排+乘积量化压缩）：512 维→PQ 编码 32 字节，内存从 1M×512×4B=2GB 压到 32MB，检索亚线性；top-k 候选回精确余弦重排。十万级以下暴力比对即可。

**Q6: ONNX int8 量化后 embedding 变化如何验证？**
A: 本工程 `optimize.py` 自带验证：同输入量化前后 embedding 余弦 >0.99 才可用；同时抽样验证阈值迁移后 FAR/FRR 变化，不能只看单点相似度。
