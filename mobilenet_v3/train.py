"""MobileNetV3 (torchvision) 训练：标准训练循环 + AMP 半精度 + 冻结微调。

MobileNetV3-Small: ~2.5M 参数; Large: ~5.4M
数据: ImageFolder 目录结构
    data/imagenet_like/ ├─ class_a/ xxx.jpg
                        └─ class_b/ xxx.jpg
未提供数据时用合成数据跑通流程。

用法:
    python -m mobilenet_v3.train --data data/imagenet_like --epochs 10
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_dataset(root: str, imgsz: int = 160):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    if os.path.isdir(root):
        ds = datasets.ImageFolder(root, transform=train_tf)
        return ds, len(ds.classes)
    print("[WARN] 未提供数据目录，使用合成数据跑通流程（替换成 ImageFolder 即真实训练）")
    # 伪数据: 4 类随机图
    data = torch.randn(128, 3, imgsz, imgsz)
    labels = torch.randint(0, 4, (128,))
    return torch.utils.data.TensorDataset(data, labels), 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/imagenet_like")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--variant", choices=["small", "large"], default="small")
    parser.add_argument("--freeze", action="store_true", help="只训练分类头（小内存技巧）")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=0, help="macOS 建议先用 0；数据量大时试 2/4")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="mobilenetv3.pt")
    args = parser.parse_args()

    from common.utils import get_device, seed_everything

    seed_everything(args.seed)
    device = get_device(args.device)

    if args.variant == "small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    ds, n_classes = build_dataset(args.data, args.imgsz)
    if n_classes != 1000:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    if args.freeze:
        for p in model.features.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[INFO] {args.variant}: total={total / 1e6:.1f}M, trainable={trainable / 1e6:.1f}M, classes={n_classes}, device={device}"
    )

    model = model.to(device)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        if args.freeze:
            model.features.eval()  # 冻结时同时固定 BatchNorm 统计量
        t0, total_loss, correct, count = time.time(), 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            # AMP: M 芯片上 fp16 autocast 有效（不需要 GradScaler）
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "mps"
            ):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        sched.step()
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total_loss / count:.4f} "
            f"acc={correct / count:.3f} ({time.time() - t0:.1f}s)"
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": ds.classes if hasattr(ds, "classes") else None,
            "config": vars(args),
        },
        args.out,
    )
    print(f"[DONE] saved -> {args.out}")


if __name__ == "__main__":
    main()
