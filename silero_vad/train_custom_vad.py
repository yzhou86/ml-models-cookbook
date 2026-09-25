"""自定义小 VAD 训练：当业务域与 Silero 不匹配时，训练自己的轻量 VAD。

模型: 40 维 log-mel + 4 层 1D-CNN（~100K 参数，比 Silero 还小）
数据: 两个目录 speech/ 与 noise/ 各放若干 wav（16kHz）
      不提供数据时会生成合成数据跑通流程

用法:
    python -m silero_vad.train_custom_vad --speech_dir data/speech --noise_dir data/noise
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVAD(nn.Module):
    """帧级二分类: (B, T, 40) -> (B, T) 语音概率。"""

    def __init__(self, n_mels: int = 40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):  # x: (B, T, n_mels)
        h = self.conv(x.transpose(1, 2))       # (B, C, T')
        h = h.permute(0, 2, 1)
        return self.head(h).squeeze(-1)        # (B, T')


def logmel(wav: np.ndarray, sr=16000, n_mels=40, frame=400, hop=160):
    """简易 log-mel（避免依赖 librosa 的训练路径）。"""
    frames = 1 + max(len(wav) - frame, 0) // hop
    win = np.hanning(frame).astype(np.float32)
    spec = np.empty((frames, frame // 2 + 1), dtype=np.float32)
    for i in range(frames):
        seg = wav[i * hop: i * hop + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        spec[i] = np.abs(np.fft.rfft(seg * win)) ** 2
    mel_fb = np.random.RandomState(0).rand(n_mels, spec.shape[-1]).astype(np.float32) * 0.01 + 0.1
    mel = spec @ mel_fb.T
    return np.log(mel + 1e-8)


def build_dataset(speech_dir=None, noise_dir=None, n_per_class=64, sec=2.0, sr=16000):
    """有目录用真实数据，否则合成数据（跑通用）。"""
    xs, ys = [], []
    rng = np.random.RandomState(0)
    n_frames = int(sec * sr / 160)

    def add(wav, label):
        for _ in range(max(1, n_per_class // max(len(wav) // int(sr * sec), 1))):
            start = rng.randint(0, max(len(wav) - int(sr * sec), 1))
            seg = wav[start: start + int(sr * sec)]
            m = logmel(seg)
            if len(m) >= n_frames:
                xs.append(m[:n_frames])
                ys.append(np.full(n_frames, label, dtype=np.float32))

    if speech_dir and os.path.isdir(speech_dir):
        for f in sorted(os.listdir(speech_dir))[:50]:
            if f.endswith((".wav", ".flac")):
                add(sf.read(os.path.join(speech_dir, f), dtype="float32")[0], 1)
        for f in sorted(os.listdir(noise_dir))[:50]:
            if f.endswith((".wav", ".flac")):
                add(sf.read(os.path.join(noise_dir, f), dtype="float32")[0], 0)
    else:
        print("[WARN] 未提供数据目录，使用合成数据演示流程")
        t = np.linspace(0, sec, int(sr * 60), endpoint=False)
        speech = np.sin(2 * np.pi * (200 + 300 * np.sin(2 * np.pi * 3 * t)) * t) * 0.3
        add(speech, 1)
        noise = rng.randn(len(t)) * 0.05
        add(noise, 0)

    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_dir", default=None)
    parser.add_argument("--noise_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="tiny_vad.pt")
    args = parser.parse_args()

    from common.utils import get_device
    device = get_device()
    x, y = build_dataset(args.speech_dir, args.noise_dir)
    print(f"[DATA] x={tuple(x.shape)}, y={tuple(y.shape)}, device={device}")

    model = TinyVAD().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x, y = x.to(device), y.to(device)
    n = len(x)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, 16):
            idx = perm[i: i + 16]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs} loss={tot / n:.4f}")

    torch.save(model.state_dict(), args.out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[DONE] {n_params / 1e3:.0f}K params, saved -> {args.out}")


if __name__ == "__main__":
    main()
