"""ECAPA-TDNN 微调训练（声纹分类场景，冻结骨干 + AAM-Softmax 头）。

两种方式:
A. 官方 VoxCeleb recipe 从零训练（重，M5 可跑但慢）:
   git clone https://github.com/speechbrain/speechbrain
   cd recipes/VoxCeleb/SpeakerRec && python train.py hparams/train_ecapa_tdnn.yaml
B. 本脚本: 冻结预训练骨干、只训 AAM-Softmax 分类头，
   演示"预训练 embedding + 轻量 domain adaption"范式（面试高频）。

数据目录格式（16k wav）:
    data/spk/  ├─ speaker_001/ a.wav b.wav ...
               └─ speaker_002/ a.wav ...
"""
import argparse
import os
import random
from glob import glob

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Loss（ArcFace 同款思想，声纹标配）。"""

    def __init__(self, emb_dim: int, n_classes: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, emb, labels):
        # L2 归一化 -> cos(theta)
        cosine = nn.functional.linear(
            nn.functional.normalize(emb), nn.functional.normalize(self.weight)
        )
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels[:, None], 1.0)
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return logits * self.s


def load_ecapa_backbone():
    """加载预训练 ECAPA（兼容 speechbrain 1.x / 旧版），返回冻结骨干。"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    wrapper = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    return wrapper.mods["embedding_model"].eval()


def wav_to_fbank(path: str, feat) -> torch.Tensor:
    """读 wav -> 16k -> fbank 特征 (T, F)。"""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = torch.from_numpy(wav)[None]  # (1, T)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return feat(wav)  # (T, F)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spk", help="说话人目录根路径")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="ecapa_finetuned")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(__doc__)
        print("[ERROR] 请先按上面的目录结构准备数据（每人若干条 16k wav）")
        return

    from speechbrain.lobes.features import Fbank
    feat = Fbank()
    ecapa = load_ecapa_backbone()

    speakers = sorted(os.listdir(args.data))
    speaker2idx = {s: i for i, s in enumerate(speakers)}
    files = [(f, speaker2idx[d]) for d in speakers
             for f in glob(os.path.join(args.data, d, "*.wav"))]
    random.shuffle(files)
    print(f"[DATA] {len(speakers)} speakers, {len(files)} wavs")

    head = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Linear(192, 192))
    loss_fn = AAMSoftmax(192, len(speakers))
    params = list(head.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    for epoch in range(args.epochs):
        total, correct, count = 0.0, 0, 0
        for path, label in files:
            feats = wav_to_fbank(path, feat)            # (T, F)
            with torch.no_grad():
                emb = ecapa(feats[None].float())        # (1, 192) 骨干冻结
            emb = head(emb)
            logits = loss_fn(emb, torch.tensor([label]))
            loss = nn.functional.cross_entropy(logits, torch.tensor([label]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            correct += (logits.argmax(-1).item() == label)
            count += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / count:.4f} acc={correct / count:.3f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"head": head.state_dict(), "aam": loss_fn.state_dict(),
                "speakers": speakers}, os.path.join(args.out, "finetune.pt"))
    print(f"[DONE] 微调头已保存 -> {args.out}/finetune.pt")
    print("[NOTE] 预训练 ECAPA embedding 直接用即可；微调头用于业务说话人分类")


if __name__ == "__main__":
    main()
