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
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common.utils import (
    benchmark,
    download,
    get_device,
    get_qengine,
    load_audio_mono,
    model_size_mb,
    print_bench,
)
from whisper_asr.infer import resolve_model_name

SAMPLE_WAV = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"


def load_inputs(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    speech, sr = load_audio_mono(download(SAMPLE_WAV, "jfk.flac"), target_sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    return processor, inputs.input_features, inputs.attention_mask


@torch.no_grad()
def run_decode(model, input_features, attention_mask, max_new_tokens=20):
    return model.generate(input_features, attention_mask=attention_mask, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    model_name = resolve_model_name(args.model)
    print(f"[INFO] qengine = {get_qengine()}")
    processor, input_features, attention_mask = load_inputs(model_name)

    # ---- 1. fp32 基线 (CPU) ----
    model_fp32 = WhisperForConditionalGeneration.from_pretrained(model_name).eval()
    model_fp32.generation_config.language = "en"
    model_fp32.generation_config.task = "transcribe"

    def decode(model, features, mask=attention_mask):
        return run_decode(model, features, mask, args.max_new_tokens)

    ids_fp32 = decode(model_fp32, input_features)
    print_bench("fp32 (CPU)", benchmark(lambda: decode(model_fp32, input_features), 1, args.runs))

    # ---- 2. 动态 int8 量化 (Linear 层权重 int8) ----
    model_int8 = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model_fp32), {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    ids_int8 = decode(model_int8, input_features)
    print_bench("int8 dynamic (CPU)", benchmark(lambda: decode(model_int8, input_features), 1, args.runs))
    print(f"[SIZE] fp32={model_size_mb(model_fp32):.1f}MB -> int8={model_size_mb(model_int8):.1f}MB")

    # ---- 3. MPS 浮点与 torch.compile（首次编译可能需要数分钟） ----
    device = get_device()
    if device.type == "mps":
        model_mps = copy.deepcopy(model_fp32).to(device)
        features_mps = input_features.to(device)
        mask_mps = attention_mask.to(device)
        print_bench(
            "fp32 (MPS)",
            benchmark(
                lambda: decode(model_mps, features_mps, mask_mps),
                1,
                args.runs,
                device=device,
            ),
        )
        if not args.skip_compile:
            try:
                compiled = torch.compile(model_mps, mode="reduce-overhead")
                decode(compiled, features_mps, mask_mps)
                print_bench(
                    "torch.compile (MPS)",
                    benchmark(
                        lambda: decode(compiled, features_mps, mask_mps),
                        1,
                        args.runs,
                        device=device,
                    ),
                )
            except Exception as e:
                print(f"[WARN] torch.compile(MPS) 跳过: {e}")
    else:
        print("[INFO] 当前不是 MPS 环境，跳过 MPS/compile 对比")

    # ---- 输出一致性检查 ----
    for name, ids in [("fp32", ids_fp32), ("int8", ids_int8)]:
        print(f"[{name}] -> {processor.decode(ids[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
