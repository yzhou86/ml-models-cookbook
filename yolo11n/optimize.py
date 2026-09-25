"""YOLO11n 优化加速：ONNX 导出 + ONNX Runtime 推理 + int8 动态量化 + CoreML 导出。

面试要点:
1. ultralytics 一行导出: model.export(format='onnx'|'coreml'|'tflite'...)
2. ONNX Runtime 动态量化: 权重 int8, 精度损失通常 <1 mAP
3. macOS 原生部署可用 CoreML（M 芯片 GPU 加速）
"""
import argparse
import os


def export_and_benchmark(weights: str, imgsz: int, runs: int):
    from ultralytics import YOLO
    model = YOLO(weights)

    # ---- 1. PyTorch 基线 ----
    from common.utils import benchmark, download, print_bench
    src = download("https://ultralytics.com/images/bus.jpg", "bus.jpg")
    model.predict(src, imgsz=imgsz, device="cpu")  # warmup
    print_bench("YOLO11n PyTorch (CPU)", benchmark(lambda: model.predict(src, imgsz=imgsz, device="cpu", verbose=False), 2, runs))

    # ---- 2. ONNX 导出 ----
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True)
    print(f"[EXPORT] ONNX -> {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f}MB)")

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    def run_onnx():
        sess.run(None, {inp.name: dummy})

    run_onnx()
    print_bench("YOLO11n ONNX Runtime (CPU)", benchmark(run_onnx, 3, runs))

    # ---- 3. ONNX int8 动态量化 ----
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = str(onnx_path).replace(".onnx", "_int8.onnx")
    quantize_dynamic(str(onnx_path), int8_path, weight_type=QuantType.QInt8)
    sess_q = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    run_int8 = lambda: sess_q.run(None, {sess_q.get_inputs()[0].name: dummy})
    run_int8()
    print_bench("YOLO11n ONNX int8 (CPU)", benchmark(run_int8, 3, runs))
    print(f"[EXPORT] int8 ONNX -> {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f}MB, 缩到 ~1/4)")

    # ---- 4. CoreML 导出（macOS 原生，可选） ----
    try:
        coreml_path = model.export(format="coreml", imgsz=imgsz, nms=True)
        print(f"[EXPORT] CoreML -> {coreml_path}（可在 Mac/iOS 上用 Xcode 或 coremltools 加载）")
    except Exception as e:
        print(f"[WARN] CoreML 导出跳过: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    export_and_benchmark(args.weights, args.imgsz, args.runs)


if __name__ == "__main__":
    main()
