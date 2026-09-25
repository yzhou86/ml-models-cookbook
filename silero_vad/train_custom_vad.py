"""自定义小 VAD 训练：当业务域与 Silero 不匹配时，训练自己的轻量 VAD。

模型: 40 维 log-mel + 4 层空洞 1D-CNN（~75K 参数，比 Silero 还小）
数据: 两个目录 speech/ 与 noise/ 各放若干 wav（16kHz）
      不提供数据时会生成合成数据跑通流程

用法:
    python -m silero_vad.train_custom_vad --speech_dir data/speech --noise_dir data/noise
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVAD(nn.Module):
    """帧级二分类: (B, T, 40) -> (B, T) 语音概率。"""

    def __init__(self, n_mels: int = 40, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=8, dilation=4),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=16, dilation=8),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):  # x: (B, T, n_mels)
        h = self.conv(x.transpose(1, 2))  # (B, C, T')
        h = h.permute(0, 2, 1)
        return self.head(h).squeeze(-1)  # (B, T')


def logmel(wav: np.ndarray, sr=16000, n_mels=40, frame=400, hop=160):
    """真实 Mel 滤波器组特征，输出 (T, n_mels)，关闭居中填充便于流式对齐。"""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sr,
        n_fft=512,
        win_length=frame,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
        center=False,
    )
    return np.log(np.maximum(mel, 1e-8)).T.astype(np.float32)


def build_dataset(speech_dir=None, noise_dir=None, n_per_class=64, sec=2.0, sr=16000):
    """有两个目录时采样真实数据，否则生成可重复的合成数据。"""
    xs, ys = [], []
    rng = np.random.RandomState(0)
    segment_samples = int(sec * sr)

    def crop_or_pad(wav):
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < segment_samples:
            return np.pad(wav, (0, segment_samples - len(wav)))
        start = rng.randint(0, len(wav) - segment_samples + 1)
        return wav[start : start + segment_samples]

    def add(wav, label):
        feature = logmel(crop_or_pad(wav), sr=sr)
        xs.append(feature)
        ys.append(np.full(len(feature), label, dtype=np.float32))

    have_speech = bool(speech_dir and os.path.isdir(speech_dir))
    have_noise = bool(noise_dir and os.path.isdir(noise_dir))
    if have_speech != have_noise:
        raise ValueError("真实数据训练必须同时提供有效的 --speech-dir 和 --noise-dir")

    if have_speech and have_noise:
        from common.utils import load_audio_mono

        def load_pool(root):
            paths = [
                os.path.join(root, f)
                for f in sorted(os.listdir(root))
                if f.lower().endswith((".wav", ".flac"))
            ]
            if not paths:
                raise ValueError(f"目录中没有 wav/flac: {root}")
            return [load_audio_mono(p, target_sr=sr)[0] for p in paths[:50]]

        speech_pool, noise_pool = load_pool(speech_dir), load_pool(noise_dir)
        for i in range(n_per_class):
            add(speech_pool[i % len(speech_pool)], 1)
            add(noise_pool[i % len(noise_pool)], 0)
    else:
        print("[WARN] 未提供数据目录，使用合成数据演示流程")
        t = np.arange(segment_samples, dtype=np.float32) / sr
        for _ in range(n_per_class):
            f0 = rng.uniform(100, 260)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t)
            speech = envelope * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t))
            speech = speech.astype(np.float32) * 0.15 + rng.randn(segment_samples).astype(np.float32) * 0.01
            noise = rng.randn(segment_samples).astype(np.float32) * rng.uniform(0.02, 0.12)
            add(speech, 1)
            add(noise, 0)

    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-dir", default=None)
    parser.add_argument("--noise-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="tiny_vad.pt")
    args = parser.parse_args()

    from common.utils import get_device, seed_everything

    seed_everything(args.seed)
    device = get_device()
    x, y = build_dataset(
        args.speech_dir, args.noise_dir, n_per_class=args.samples_per_class, sec=args.clip_seconds
    )
    print(f"[DATA] x={tuple(x.shape)}, y={tuple(y.shape)}, device={device}")

    model = TinyVAD(channels=args.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x, y = x.to(device), y.to(device)
    n = len(x)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs} loss={tot / n:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_mels": 40,
                "channels": args.channels,
                "sample_rate": 16000,
                "frame": 400,
                "hop": 160,
            },
        },
        args.out,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[DONE] {n_params / 1e3:.0f}K params, saved -> {args.out}")


if __name__ == "__main__":
    main()
