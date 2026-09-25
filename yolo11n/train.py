"""YOLO11n 训练（ultralytics，M 芯片走 MPS）。

coco8 是官方 8 图玩具数据集，专用于跑通流程；
实际项目替换 data=yaml 路径（如 coco128.yaml 或自定义）。

用法:
    python -m yolo11n.train                       # coco8 微调演示
    python -m yolo11n.train --data coco128.yaml --epochs 30
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=320, help="M5 上先用小分辨率练手")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "auto"], default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0, help="macOS 建议 0；稳定后可试 2/4")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze", type=int, default=0, help="冻结前 N 层，0 表示不冻结")
    args = parser.parse_args()

    from ultralytics import YOLO

    from common.utils import get_device

    resolved = get_device(args.device)
    device = 0 if resolved.type == "cuda" else resolved.type
    print(f"[INFO] train device = {device}")

    model = YOLO(args.weights)  # yolo11n: 6.5MB 权重, ~3M 参数
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        freeze=args.freeze or None,
        verbose=True,
    )
    print(f"[DONE] 训练完成, 输出目录={results.save_dir}")
    metrics = {key: round(float(value), 5) for key, value in results.results_dict.items()}
    print(f"[METRICS] {metrics}")


if __name__ == "__main__":
    main()
