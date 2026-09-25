"""YOLO11n 优化：公平的端到端基准、ONNX 静态 PTQ 与可选 CoreML 导出。"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def image_paths(calib_dir: str | None, fallback: str, limit: int) -> list[str]:
    if not calib_dir:
        print("[WARN] 未提供 --calib-dir，仅用示例图做 PTQ smoke test；不能代表真实 mAP")
        return [fallback]
    paths = [
        str(p)
        for p in sorted(Path(calib_dir).rglob("*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ][:limit]
    if not paths:
        raise ValueError(f"校准目录中没有图片: {calib_dir}")
    return paths


def preprocess_yolo(path: str, imgsz: int) -> np.ndarray:
    """复用 Ultralytics LetterBox，生成导出 ONNX 所需的 NCHW RGB float32。"""
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    image = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(image[None], dtype=np.float32) / 255.0


def quantize_onnx(fp32_path: str, int8_path: str, paths: list[str], imgsz: int) -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    input_name = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    graph = onnx.load(fp32_path)
    # 检测头含 DFL、sigmoid 和最终坐标解码，对量化范围非常敏感；保留为 fp32。
    head_nodes = [node.name for node in graph.graph.node if node.name.startswith("/model.23")]
    print(f"[QUANT] keep detection head in fp32, excluded nodes={len(head_nodes)}")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.data = iter([{input_name: preprocess_yolo(p, imgsz)} for p in paths])

        def get_next(self):
            return next(self.data, None)

    quantize_static(
        fp32_path,
        int8_path,
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=head_nodes,
        extra_options={"WeightSymmetric": True},
    )


def prediction_summary(result) -> str:
    boxes = result[0].boxes
    if len(boxes) == 0:
        return "0 boxes"
    best = int(boxes.conf.argmax().item())
    cls = int(boxes.cls[best].item())
    return f"{len(boxes)} boxes, best={result[0].names[cls]}({boxes.conf[best].item():.3f})"


def benchmark_predict(model, src: str, imgsz: int, runs: int, device: str, title: str):
    from common.utils import benchmark, print_bench

    def run():
        return model.predict(src, imgsz=imgsz, device=device, verbose=False, save=False)

    result = run()
    print_bench(title, benchmark(run, 2, runs, device=device if device in {"mps", "cuda"} else None))
    print(f"[CHECK] {title}: {prediction_summary(result)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--calib-samples", type=int, default=100)
    parser.add_argument("--rebuild-int8", action="store_true")
    parser.add_argument("--coreml", action="store_true", help="额外导出 CoreML（需 uv sync --extra apple）")
    args = parser.parse_args()

    from ultralytics import YOLO

    from common.utils import download, get_device

    src = download("https://ultralytics.com/images/bus.jpg", "bus.jpg")
    model = YOLO(args.weights)

    # 同样的图片、分辨率、预处理、解码和 NMS，比较端到端延迟。
    benchmark_predict(model, src, args.imgsz, args.runs, "cpu", "YOLO11n PyTorch CPU e2e")
    device = get_device()
    if device.type == "mps":
        benchmark_predict(model, src, args.imgsz, args.runs, "mps", "YOLO11n PyTorch MPS e2e")

    onnx_path = str(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True))
    print(f"[EXPORT] ONNX -> {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f}MB)")
    onnx_model = YOLO(onnx_path, task="detect")
    benchmark_predict(onnx_model, src, args.imgsz, args.runs, "cpu", "YOLO11n ONNX CPU e2e")

    int8_path = str(Path(onnx_path).with_name(Path(onnx_path).stem + "_int8.onnx"))
    if args.rebuild_int8 or not os.path.exists(int8_path):
        paths = image_paths(args.calib_dir, src, args.calib_samples)
        print(f"[QUANT] QDQ static PTQ, calibration images={len(paths)}")
        quantize_onnx(onnx_path, int8_path, paths, args.imgsz)
    print(f"[EXPORT] int8 ONNX -> {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f}MB)")
    try:
        int8_model = YOLO(int8_path, task="detect")
        benchmark_predict(int8_model, src, args.imgsz, args.runs, "cpu", "YOLO11n ONNX int8 e2e")
    except Exception as exc:
        print(f"[WARN] 当前 ONNX Runtime 不支持该量化图，保留 fp32 ONNX: {exc}")

    if args.coreml:
        try:
            coreml_path = model.export(format="coreml", imgsz=args.imgsz, nms=True, half=True)
            print(f"[EXPORT] CoreML fp16 -> {coreml_path}")
        except Exception as exc:
            print(f"[WARN] CoreML 导出失败: {exc}")


if __name__ == "__main__":
    main()
