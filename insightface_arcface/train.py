"""ArcFace 人脸识别从零/迁移训练（纯 PyTorch 实现，面试讲原理用）。

核心思想: embedding + Additive Angular Margin Loss
  - 骨干: MobileNetV2 轻量骨干（M5 可训）
  - 损失: cos(theta + m) 加角间距 -> 类内更紧、类间更开
数据目录格式:
    data/faces/ ├─ person_001/ a.jpg b.jpg
                └─ person_002/ a.jpg ...
"""

import argparse
import math
import os

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
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.margin_correction = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        target = cosine * self.cos_m - sine * self.sin_m
        # 保持 cos(theta+m) 在 [0, pi] 外仍单调，避免困难样本梯度反向。
        target = torch.where(cosine > self.threshold, target, cosine - self.margin_correction)
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
    tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="arcface_student.pt")
    args = parser.parse_args()

    from common.utils import get_device, seed_everything

    seed_everything(args.seed)
    device = get_device(args.device)

    ds = build_dataset(args.data)
    n_classes = len(ds.classes) if isinstance(ds, datasets.ImageFolder) else 8
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"[DATA] {len(ds)} images, {n_classes} classes, device={device}")

    model = FaceNet().to(device)
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    arcface = ArcMarginProduct(128, n_classes, s=args.scale, m=args.margin).to(device)
    params = [p for p in [*model.parameters(), *arcface.parameters()] if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            model.features.eval()
        total, correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            # ArcFace 的 acos/cos 对半精度更敏感；embedding 后强制回 fp32 计算 margin。
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "mps"
            ):
                emb = model(x)
            logits = arcface(emb.float(), y)
            with torch.autocast(device_type=device.type, enabled=False):
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / count:.4f} acc={correct / count:.3f}")

    torch.save(
        {
            "model": model.state_dict(),
            "arcface": arcface.state_dict(),
            "classes": ds.classes if isinstance(ds, datasets.ImageFolder) else None,
            "config": vars(args),
        },
        args.out,
    )
    print(f"[DONE] saved -> {args.out}")
    print("""
[SUMMARY] ArcFace 面试要点:
1. softmax 类内不紧 -> 角度间隔 m 拉开决策边界
2. s 缩放归一化后的 logits；ArcFace m 常从 0.3~0.5 扫描，CosFace 常用约 0.35
3. 工业落地: 检测(SCRFD) -> 对齐(5 点) -> embedding -> 余弦比对
""")


if __name__ == "__main__":
    main()
