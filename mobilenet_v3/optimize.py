"""MobileNetV3-Large 优化：动态量化、静态 PTQ、MPS 与 torch.compile。

静态 PTQ 使用 torchvision 的 QuantizableMobileNetV3，而不是普通浮点模型。
默认用示例图片完成 smoke calibration；正式精度评估应传入真实校准图片目录。
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from common.utils import benchmark, download, get_device, get_qengine, model_size_mb, print_bench


def calibration_batches(preprocess, sample_path: str, calib_dir: str | None, limit: int):
    """构造 batch=1 校准输入；生产 PTQ 应使用与部署分布一致的无标签图片。"""
    paths = []
    if calib_dir:
        root = Path(calib_dir)
        paths = [
            p for p in sorted(root.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ][:limit]
        if not paths:
            raise ValueError(f"校准目录中没有图片: {calib_dir}")
    else:
        paths = [Path(sample_path)]
        print("[WARN] 未提供 --calib-dir，仅做单图校准演示，不能据此评价 PTQ 精度")
    return [preprocess(Image.open(p).convert("RGB"))[None] for p in paths]


def build_ptq_model(fp32: nn.Module) -> nn.Module:
    """将浮点权重装入 torchvision 可量化结构，再执行 eager-mode PTQ。"""
    from torchvision.models.quantization import mobilenet_v3_large as quantizable_mobilenet_v3_large

    model = quantizable_mobilenet_v3_large(weights=None, quantize=False).eval()
    model.load_state_dict(fp32.state_dict())
    model.fuse_model(is_qat=False)
    model.qconfig = torch.ao.quantization.get_default_qconfig(get_qengine())
    return torch.ao.quantization.prepare(model, inplace=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--calib-dir", default=None, help="真实校准图片目录；建议 100~500 张代表性图片")
    parser.add_argument("--calib-samples", type=int, default=100)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] qengine={get_qengine()}")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    fp32 = models.mobilenet_v3_large(weights=weights).eval()
    preprocess = weights.transforms()
    sample_path = download("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    calibration = calibration_batches(preprocess, sample_path, args.calib_dir, args.calib_samples)
    sample = calibration[0]

    print_bench("MobileNetV3-L fp32 CPU", benchmark(lambda: fp32(sample), 3, args.runs))

    # 动态量化只覆盖 Linear，适合作为“为什么 CNN 收益有限”的对照组。
    dynamic = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    print_bench("MobileNetV3-L dynamic CPU", benchmark(lambda: dynamic(sample), 3, args.runs))

    prepared = build_ptq_model(fp32)
    with torch.inference_mode():
        for batch in calibration:
            prepared(batch)
    ptq = torch.ao.quantization.convert(prepared, inplace=False).eval()
    print_bench("MobileNetV3-L static PTQ", benchmark(lambda: ptq(sample), 3, args.runs))

    with torch.inference_mode():
        fp_logits = fp32(sample)
        ptq_logits = ptq(sample)
        fp_top1 = int(fp_logits.argmax(1).item())
        ptq_top1 = int(ptq_logits.argmax(1).item())
        cosine = torch.nn.functional.cosine_similarity(fp_logits, ptq_logits).item()
    print(
        f"[CHECK] sample top1: fp32={weights.meta['categories'][fp_top1]!r}, "
        f"ptq={weights.meta['categories'][ptq_top1]!r}, logits cosine={cosine:.4f}"
    )
    print(
        f"[SIZE] fp32={model_size_mb(fp32):.1f}MB, dynamic={model_size_mb(dynamic):.1f}MB, "
        f"static={model_size_mb(ptq):.1f}MB"
    )

    device = get_device()
    if device.type == "mps":
        model_mps = copy.deepcopy(fp32).to(device)
        sample_mps = sample.to(device)
        print_bench(
            "MobileNetV3-L fp32 MPS", benchmark(lambda: model_mps(sample_mps), 5, args.runs, device=device)
        )
        if not args.skip_compile:
            try:
                compiled = torch.compile(model_mps, mode="reduce-overhead")
                compiled(sample_mps)
                print_bench(
                    "MobileNetV3-L compile MPS",
                    benchmark(lambda: compiled(sample_mps), 3, args.runs, device=device),
                )
            except Exception as exc:
                print(f"[WARN] torch.compile(MPS) 跳过: {exc}")

    print("""
[SUMMARY]
- 动态量化只处理分类头，CNN 通常收益有限。
- 静态 PTQ 会量化卷积和激活，校准集质量决定精度。
- MPS 对 batch=1 小模型未必快于 CPU；应以本机实测为准。
""")


if __name__ == "__main__":
    main()
