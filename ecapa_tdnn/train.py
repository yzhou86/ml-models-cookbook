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
import math
import os
import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Loss（ArcFace 同款思想，声纹标配）。"""

    def __init__(self, emb_dim: int, n_classes: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.margin_correction = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        # L2 归一化 -> cos(theta)
        cosine = nn.functional.linear(nn.functional.normalize(emb), nn.functional.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        target_logits = cosine * self.cos_m - sine * self.sin_m
        target_logits = torch.where(cosine > self.threshold, target_logits, cosine - self.margin_correction)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels[:, None], 1.0)
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return logits * self.s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spk", help="说话人目录根路径")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="仅分类头使用该设备；ECAPA 特征提取固定走 CPU",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="ecapa_finetuned")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(__doc__)
        print("[ERROR] 请先按上面的目录结构准备数据（每人若干条 16k wav）")
        return

    from common.utils import get_device, seed_everything

    seed_everything(args.seed)
    device = get_device(args.device)
    from ecapa_tdnn.infer import encode_file, load_ecapa

    ecapa = load_ecapa("cpu")

    speakers = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    speaker2idx = {s: i for i, s in enumerate(speakers)}
    files = [
        (f, speaker2idx[d])
        for d in speakers
        for pattern in ("*.wav", "*.flac")
        for f in glob(os.path.join(args.data, d, pattern))
    ]
    random.shuffle(files)
    if len(speakers) < 2 or not files:
        raise ValueError("至少需要 2 个说话人目录，且目录中包含 wav/flac 文件")
    print(f"[DATA] {len(speakers)} speakers, {len(files)} files; 预计算冻结骨干 embedding")

    # 骨干冻结时 embedding 与 epoch 无关，只计算一次可显著缩短训练时间。
    embeddings, labels = [], []
    with torch.inference_mode():
        for path, label in files:
            emb = encode_file(ecapa, path).reshape(-1, 192)[0]
            embeddings.append(emb.cpu())
            labels.append(label)
    dataset = TensorDataset(torch.stack(embeddings), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    head = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Linear(192, 192)).to(device)
    loss_fn = AAMSoftmax(192, len(speakers), s=args.scale, m=args.margin).to(device)
    params = list(head.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(args.epochs):
        total, correct, count = 0.0, 0, 0
        for emb, label in loader:
            emb, label = emb.to(device), label.to(device)
            emb = head(emb)
            logits = loss_fn(emb, label)
            loss = nn.functional.cross_entropy(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(label)
            correct += (logits.argmax(-1) == label).sum().item()
            count += len(label)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / count:.4f} acc={correct / count:.3f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(
        {
            "head": head.cpu().state_dict(),
            "aam": loss_fn.cpu().state_dict(),
            "speakers": speakers,
            "config": {"margin": args.margin, "scale": args.scale},
        },
        os.path.join(args.out, "finetune.pt"),
    )
    print(f"[DONE] 微调头已保存 -> {args.out}/finetune.pt")
    print("[NOTE] 预训练 ECAPA embedding 直接用即可；微调头用于业务说话人分类")


if __name__ == "__main__":
    main()
