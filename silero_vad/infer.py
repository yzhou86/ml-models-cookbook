"""Silero VAD 推理：检测语音片段的时间戳（约 2M 参数，极致轻量）。

用法:
    python -m silero_vad.infer
"""
import torch


def load_silero_vad():
    """加载官方 Silero VAD（JITScript，~2MB）。"""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model, utils


def get_timestamps(speech_prob: "list[tuple[float, float]]", threshold=0.5, min_gap=0.1):
    """把逐窗概率转成 [start, end] 时间戳。"""
    segments, start = [], None
    for t, p in speech_prob:
        if p >= threshold and start is None:
            start = t
        elif p < threshold and start is not None:
            if t - start >= min_gap:
                segments.append((round(start, 2), round(t, 2)))
            start = None
    if start is not None:
        segments.append((round(start, 2), round(speech_prob[-1][0], 2)))
    return segments


def main():
    model, utils = load_silero_vad()
    (get_speech_timestamps, save_audio, read_audio, _, _) = utils

    from common.utils import download
    wav_path = download(
        "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav", "preamble10.wav"
    )
    wav = read_audio(wav_path, sampling_rate=16000)  # -> 1D float tensor

    # 方式 1: 官方 API
    stamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    print("[RESULT] speech segments (samples):", stamps[:5], "...")

    # 方式 2: 手动滑窗逐帧取概率（面试可讲原理: 512 样本窗口 + 单向 LSTM）
    window = wav
    model.reset_states()
    probs, hop = [], 512
    with torch.no_grad():
        for i in range(0, max(len(window) - hop, 1), hop):
            chunk = window[i: i + hop]
            if len(chunk) < hop:
                chunk = torch.nn.functional.pad(chunk, (0, hop - len(chunk)))
            p = model(chunk, 16000).item()
            probs.append((i / 16000, p))
    segments = get_timestamps(probs)
    print(f"[RESULT] 手动滑窗 -> {len(segments)} 段语音: {segments}")

    # 逐段裁剪保存
    import soundfile as sf
    arr = window.numpy()
    for idx, (s, e) in enumerate(segments[:3]):
        sf.write(f"samples/vad_seg{idx}.wav", arr[int(s * 16000): int(e * 16000)], 16000)
    print("[DONE] 前 3 段已保存到 samples/vad_seg*.wav")


if __name__ == "__main__":
    main()
