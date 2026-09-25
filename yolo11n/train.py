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
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 训练")
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else ("cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[INFO] train device = {device}")

    model = YOLO(args.weights)  # yolo11n: 6.5MB 权重, ~3M 参数
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        device=device,
        workers=0,      # macOS spawn 问题规避
        patience=3,
        verbose=True,
    )
    print(f"[DONE] 训练完成, best=runs/detect/train/weights/best.pt")
    print(f"[INFO] results: {results}")


if __name__ == "__main__":
    main()
