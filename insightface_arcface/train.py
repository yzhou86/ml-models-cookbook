"""ArcFace 人脸识别从零/迁移训练（纯 PyTorch 实现，面试讲原理用）。

核心思想: embedding + Additive Angular Margin Loss
  - 骨干: MobileNetV2 轻量骨干（M5 可训）
  - 损失: cos(theta + m) 加角间距 -> 类内更紧、类间更开
数据目录格式:
    data/faces/ ├─ person_001/ a.jpg b.jpg
                └─ person_002/ a.jpg ...
"""
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class ArcMarginProduct(nn.Module):
    """ArcFace loss 实现（s=64, m=0.5 为官方推荐）。"""

    def __init__(self, in_features: int, out_features: int, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target * one_hot
        return logits * self.s


class FaceNet(nn.Module):
    """轻量人脸 embedding 网络: 骨干 + BN + dropout + 128 维嵌入。"""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(1280)
        self.dropout = nn.Dropout(0.3)
        self.emb = nn.Linear(1280, emb_dim)

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.emb(self.dropout(self.bn(h)))


def build_dataset(root: str):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if os.path.isdir(root):
        return datasets.ImageFolder(root, transform=tf)
    print("[WARN] 未提供数据目录，使用合成数据跑通流程")
    data = torch.randn(64, 3, 112, 112)
    labels = torch.randint(0, 8, (64,))
    return torch.utils.data.TensorDataset(data, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/faces")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="arcface_student.pt")
    args = parser.parse_args()

    from common.utils import get_device
    device = get_device()

    ds = build_dataset(args.data)
    n_classes = len(ds.classes) if isinstance(ds, datasets.ImageFolder) else 8
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"[DATA] {len(ds)} images, {n_classes} classes, device={device}")

    model = FaceNet().to(device)
    arcface = ArcMarginProduct(128, n_classes).to(device)
    opt = torch.optim.AdamW([*model.parameters(), *arcface.parameters()], lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total, correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "mps"):
                emb = model(x)
                logits = arcface(emb, y)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / count:.4f} acc={correct / count:.3f}")

    torch.save({"model": model.state_dict(), "arcface": arcface.state_dict()}, args.out)
    print(f"[DONE] saved -> {args.out}")
    print("""
[SUMMARY] ArcFace 面试要点:
1. softmax 类内不紧 -> 角度间隔 m 拉开决策边界
2. s 固定特征范数, m 一般 0.5(ArcFace 加性间隔) / 1.35(CosFace 余弦间隔变体)
3. 工业落地: 检测(SCRFD) -> 对齐(5 点) -> embedding -> 余弦比对
""")


if __name__ == "__main__":
    main()
