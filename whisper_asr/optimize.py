"""Whisper 优化加速对比: fp32 vs 动态 int8 量化 vs torch.compile。

量化模型只能在 CPU 运行（qnnpack 后端），本脚本统一在 CPU 上对比公平。
面试要点:
1. 动态量化: nn.Linear 权重 int8、激活动态量化，无需校准数据
2. torch.compile: Inductor 算子融合 + kernel 优化
3. faster-whisper(CTranslate2) 在 macOS 上仅支持 CPU，思路同 int8 量化
"""
import argparse
import copy

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common.utils import benchmark, download, print_bench

SAMPLE_WAV = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"


def load_inputs(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    import soundfile as sf
    speech, sr = sf.read(download(SAMPLE_WAV, "preamble10.wav"), dtype="float32")
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    return processor, inputs.input_features


@torch.no_grad()
def run_decode(model, input_features, max_new_tokens=20):
    return model.generate(input_features, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"[INFO] qengine = {torch.backends.quantized.engine}")
    processor, input_features = load_inputs(args.model)

    # ---- 1. fp32 基线 (CPU) ----
    model_fp32 = WhisperForConditionalGeneration.from_pretrained(args.model).eval()
    ids_fp32 = run_decode(model_fp32, input_features)
    print_bench("fp32 (CPU)", benchmark(lambda: run_decode(model_fp32, input_features), 1, args.runs))

    # ---- 2. 动态 int8 量化 (Linear 层权重 int8) ----
    model_int8 = copy.deepcopy(model_fp32)
    torch.ao.quantization.quantize_dynamic(model_int8, dtype=torch.qint8)
    ids_int8 = run_decode(model_int8, input_features)
    mem_fp32 = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
    print_bench("int8 dynamic (CPU)", benchmark(lambda: run_decode(model_int8, input_features), 1, args.runs))
    print(f"[INFO] fp32 权重内存 ~{mem_fp32 / 1e6:.0f}MB -> int8 约 1/4")

    # ---- 3. torch.compile (失败自动降级) ----
    try:
        model_compiled = torch.compile(model_fp32, mode="reduce-overhead")
        run_decode(model_compiled, input_features)  # 预热触发编译
        print_bench("torch.compile (CPU)", benchmark(lambda: run_decode(model_compiled, input_features), 1, args.runs))
    except Exception as e:
        print(f"[WARN] torch.compile 不可用，降级跳过: {e}")

    # ---- 输出一致性检查 ----
    for name, ids in [("fp32", ids_fp32), ("int8", ids_int8)]:
        print(f"[{name}] -> {processor.decode(ids[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
