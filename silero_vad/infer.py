"""Silero VAD 推理：检测语音片段的时间戳（约 2M 参数，极致轻量）。

用法:
    python -m silero_vad.infer
"""

import argparse

import torch

SILERO_REVISION = "5cd7945676eb32225748052e2e6a0580e4686a08"
JIT_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/"
    "src/silero_vad/data/silero_vad.jit"
)
JIT_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
ONNX_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/"
    "src/silero_vad/data/silero_vad.onnx"
)
ONNX_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def load_silero_vad():
    """直接加载官方 JIT，避免本地 silero_vad 包名与 Torch Hub 包冲突。"""
    from common.utils import download

    path = download(JIT_URL, "silero_vad.jit", sha256=JIT_SHA256)
    return torch.jit.load(path, map_location="cpu").eval()


def get_timestamps(
    speech_prob: "list[tuple[float, float]]",
    threshold: float = 0.5,
    end_threshold: float | None = None,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.1,
    frame_duration: float = 0.032,
):
    """用迟滞阈值和静音 hangover 把逐窗概率转换为秒级时间戳。"""
    if not speech_prob:
        return []
    end_threshold = threshold - 0.15 if end_threshold is None else end_threshold
    segments, start, silence_start = [], None, None
    for t, p in speech_prob:
        if p >= threshold and start is None:
            start = t
            silence_start = None
        elif start is not None:
            if p < end_threshold:
                silence_start = t if silence_start is None else silence_start
                if t - silence_start + frame_duration >= min_silence_duration:
                    if silence_start - start >= min_speech_duration:
                        segments.append((round(start, 3), round(silence_start, 3)))
                    start, silence_start = None, None
            else:
                silence_start = None
    if start is not None:
        end = speech_prob[-1][0] + frame_duration
        if end - start >= min_speech_duration:
            segments.append((round(start, 3), round(end, 3)))
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=float, default=100)
    parser.add_argument("--min-silence-ms", type=float, default=100)
    args = parser.parse_args()

    model = load_silero_vad()

    from common.utils import SAMPLE_DIR, download, load_audio_mono

    wav_path = args.audio or download(
        "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", "jfk.flac"
    )
    wav_np, _ = load_audio_mono(wav_path, target_sr=16000)
    wav = torch.from_numpy(wav_np)

    # 手动滑窗逐帧取概率（面试可讲原理: 512 样本窗口 + 单向 LSTM）
    window = wav
    model.reset_states()
    probs, hop = [], 512
    with torch.no_grad():
        for i in range(0, len(window), hop):
            chunk = window[i : i + hop]
            if len(chunk) < hop:
                chunk = torch.nn.functional.pad(chunk, (0, hop - len(chunk)))
            p = model(chunk, 16000).item()
            probs.append((i / 16000, p))
    segments = get_timestamps(
        probs,
        threshold=args.threshold,
        min_speech_duration=args.min_speech_ms / 1000,
        min_silence_duration=args.min_silence_ms / 1000,
        frame_duration=hop / 16000,
    )
    print(f"[RESULT] 手动滑窗 -> {len(segments)} 段语音: {segments}")

    # 逐段裁剪保存
    import soundfile as sf

    arr = window.numpy()
    for idx, (s, e) in enumerate(segments[:3]):
        sf.write(SAMPLE_DIR / f"vad_seg{idx}.wav", arr[int(s * 16000) : int(e * 16000)], 16000)
    print(f"[DONE] 前 3 段已保存到 {SAMPLE_DIR}/vad_seg*.wav")


if __name__ == "__main__":
    main()
