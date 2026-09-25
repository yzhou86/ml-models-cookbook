"""Whisper tiny/base/small 语音识别推理。

用法:
    python -m whisper_asr.infer                # whisper-tiny
    python -m whisper_asr.infer --model base
    python -m whisper_asr.infer --model small  # 24GB 内存无压力

模型尺寸: tiny=39M, base=74M, small=244M
"""
import argparse

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common.utils import download, get_device

SAMPLE_WAV = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"


@torch.no_grad()
def transcribe(model_name: str = "openai/whisper-tiny", audio_path: str = None, language: str = "zh") -> str:
    device = get_device()
    print(f"[INFO] device={device}, model={model_name}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    model.generation_config.language = language

    if audio_path is None:
        audio_path = download(SAMPLE_WAV, "preamble10.wav")
    speech, sr = sf.read(audio_path, dtype="float32")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    print(f"[INFO] audio: {audio_path}, sr={sr}, {len(speech) / sr:.1f}s")

    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    predicted_ids = model.generate(input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-tiny",
                        choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"])
    parser.add_argument("--audio", default=None, help="本地 wav/flac 路径，缺省下载示例")
    parser.add_argument("--language", default="zh", help="zh / en ...")
    args = parser.parse_args()

    text = transcribe(args.model, args.audio, args.language)
    print(f"\n[RESULT] {text}")


if __name__ == "__main__":
    main()
