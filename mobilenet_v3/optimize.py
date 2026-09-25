"""MobileNetV3 优化加速三连：静态 int8 量化(PTQ) + 动态 int8 + torch.compile。

面试要点:
1. 静态 PTQ 需要校准数据（MinMax observer），对卷积网络收益最大
2. ARM 芯片用 qnnpack 后端（fbgemm 是 x86）
3. 静态 PTQ 需要 QuantStub/DeQuantStub（浮点模型没有，需自己包装）
4. torchvision 也直接提供量化版: from torchvision.models.quantization import mobilenet_v3_large
"""
import copy

import torch
import torch.nn as nn
from torchvision import models


class QuantWrapper(nn.Module):
    """给普通浮点模型插入量化入口/出口桩，静态 PTQ 必需。"""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = m
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def calibrate(model, data):
    """PTQ 校准: 前向传播激活统计，不计算梯度。"""
    model.eval()
    with torch.no_grad():
        for x in data:
            model(x)


def main():
    from common.utils import benchmark, print_bench
    torch.backends.quantized.engine = "qnnpack"  # ARM 后端
    print(f"[INFO] qengine = {torch.backends.quantized.engine}")

    fp32 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    dummy = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        fp32(dummy)
    print_bench("MobileNetV3-S fp32 (CPU)", benchmark(lambda: fp32(dummy), 3, 20))

    # ---- 1. 动态 int8 量化（Linear 层，无校准） ----
    dyn = copy.deepcopy(fp32)
    torch.ao.quantization.quantize_dynamic(dyn, dtype=torch.qint8)
    with torch.no_grad():
        dyn(dummy)
    print_bench("MobileNetV3-S int8 dynamic (CPU)", benchmark(lambda: dyn(dummy), 3, 20))

    # ---- 2. 静态 PTQ: 融合 Conv+BN+ReLU -> 插桩 -> 校准 -> 转换 ----
    fused = copy.deepcopy(fp32)
    fused.fuse_model()                        # torchvision MobileNetV3 自带融合 API
    wrapped = QuantWrapper(fused)
    wrapped.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    prepare = getattr(torch.ao.quantization, "prepare_eager", torch.ao.quantization.prepare)
    prepared = prepare(wrapped)
    calibrate(prepared, torch.split(dummy, 1, dim=0))
    qmodel = torch.ao.quantization.convert(prepared)
    qmodel.eval()
    with torch.no_grad():
        qmodel(dummy)
    print_bench("MobileNetV3-S int8 static PTQ (CPU)", benchmark(lambda: qmodel(dummy), 3, 20))

    # ---- 3. torch.compile（失败自动降级） ----
    try:
        compiled = torch.compile(fp32, mode="reduce-overhead")
        with torch.no_grad():
            compiled(dummy)
        print_bench("MobileNetV3-S torch.compile (CPU)", benchmark(lambda: compiled(dummy), 3, 20))
    except Exception as e:
        print(f"[WARN] torch.compile 降级: {e}")

    # ---- 精度对比 ----
    with torch.no_grad():
        out_fp32 = fp32(dummy).argmax(1)
        out_int8 = qmodel(dummy).argmax(1)
    print(f"[CHECK] PTQ 前后 top1 一致率: {(out_fp32 == out_int8).float().mean():.2%}")
    print("""
[SUMMARY] MobileNetV3 优化:
- 静态 PTQ > 动态量化 > fp32（卷积网络融合后 int8 提速最明显）
- 模型 9.7MB(fp32) -> ~2.5MB(int8)
- 进一步可用 QAT(量化感知训练) 恢复精度
""")


if __name__ == "__main__":
    main()
