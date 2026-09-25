"""Whisper tiny/base/small 语音识别推理。

用法:
    python -m whisper_asr.infer                # whisper-tiny
    python -m whisper_asr.infer --model base
    python -m whisper_asr.infer --model small  # 24GB 内存无压力

模型尺寸: tiny=39M, base=74M, small=244M
"""

import argparse

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common.utils import download, get_device, load_audio_mono

SAMPLE_WAV = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"


def resolve_model_name(name: str) -> str:
    return name if "/" in name else f"openai/whisper-{name}"


@torch.no_grad()
def transcribe(
    model_name: str = "openai/whisper-tiny",
    audio_path: str | None = None,
    language: str = "en",
    device_name: str = "auto",
    max_new_tokens: int = 128,
) -> str:
    model_name = resolve_model_name(model_name)
    device = get_device(device_name)
    print(f"[INFO] device={device}, model={model_name}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    if language != "auto":
        model.generation_config.language = language
    model.generation_config.task = "transcribe"

    if audio_path is None:
        audio_path = download(SAMPLE_WAV, "jfk.flac")
    speech, sr = load_audio_mono(audio_path, target_sr=16000)
    print(f"[INFO] audio: {audio_path}, sr={sr}, {len(speech) / sr:.1f}s")

    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    predicted_ids = model.generate(
        input_features, attention_mask=attention_mask, max_new_tokens=max_new_tokens
    )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="tiny/base/small，或完整 Hugging Face 模型名")
    parser.add_argument("--audio", default=None, help="本地 wav/flac 路径，缺省下载示例")
    parser.add_argument("--language", default="en", help="zh/en/...；auto 表示自动识别")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    text = transcribe(args.model, args.audio, args.language, args.device, args.max_new_tokens)
    print(f"\n[RESULT] {text}")


if __name__ == "__main__":
    main()
