"""YOLO11n 推理 + 可视化。

用法:
    python -m yolo11n.infer                     # 官方示例图 bus.jpg
    python -m yolo11n.infer --src your.jpg
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="图片/视频/目录路径")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO
    from common.utils import download

    src = args.src or download("https://ultralytics.com/images/bus.jpg", "bus.jpg")

    model = YOLO(args.weights)
    results = model.predict(src, conf=args.conf, imgsz=args.imgsz, device="mps")

    for r in results:
        names = r.names
        boxes = r.boxes
        print(f"[INFO] {r.path}: 检测到 {len(boxes)} 个目标")
        for b in boxes:
            cls = int(b.cls.item())
            print(f"  - {names[cls]}: conf={b.conf.item():.2f}, xyxy={[round(v, 1) for v in b.xyxy[0].tolist()]}")
        # 保存可视化结果
        out = r.save(filename="samples/yolo_result.jpg")
        print(f"[DONE] 可视化保存 -> samples/yolo_result.jpg")


if __name__ == "__main__":
    main()
