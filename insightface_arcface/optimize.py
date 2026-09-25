"""InsightFace ONNX 优化：动态 int8 量化 + 推理延迟对比。

buffalo_l 检测模型(det_10g)约 16MB、识别模型(w600k_r50)约 170MB，
int8 量化后约缩到 1/4，CPU 推理明显提速。
"""
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def bench(sess, feed, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        sess.run(None, feed)
    import time
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feed)
    return (time.perf_counter() - start) / n_runs * 1000


def main():
    from common.utils import print_bench
    from insightface_arcface.infer import CACHE, ensure_models

    sessions = ensure_models()
    root = os.path.join(CACHE, "buffalo_l")

    # ---- 1. 识别模型 w600k_r50 (ArcFace, 170MB -> ~45MB) ----
    rec_path = os.path.join(root, "w600k_r50.onnx")
    rec_int8 = rec_path.replace(".onnx", "_int8.onnx")
    if not os.path.exists(rec_int8):
        print("[QUANT] 量化 w600k_r50 ...")
        quantize_dynamic(rec_path, rec_int8, weight_type=QuantType.QInt8)

    aligned = np.random.randn(1, 3, 112, 112).astype(np.float32)
    rec_fp32 = sessions["w600k_r50"]
    rec_q = ort.InferenceSession(rec_int8, providers=["CPUExecutionProvider"])
    feed_fp32 = {rec_fp32.get_inputs()[0].name: aligned}
    feed_q = {rec_q.get_inputs()[0].name: aligned}

    print_bench("ArcFace fp32 (CPU)", bench(rec_fp32, feed_fp32))
    print_bench("ArcFace int8 (CPU)", bench(rec_q, feed_q))
    print(f"[SIZE] fp32={os.path.getsize(rec_path) / 1e6:.0f}MB -> int8={os.path.getsize(rec_int8) / 1e6:.0f}MB")

    # 精度: cosine 相似度对比
    e1 = rec_fp32.run(None, feed_fp32)[0][0]
    e2 = rec_q.run(None, feed_q)[0][0]
    cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    print(f"[CHECK] 量化前后 embedding cosine = {cos:.4f}")

    # ---- 2. 检测模型 det_10g 量化 ----
    det_path = os.path.join(root, "det_10g.onnx")
    det_int8 = det_path.replace(".onnx", "_int8.onnx")
    if not os.path.exists(det_int8):
        print("[QUANT] 量化 det_10g ...")
        quantize_dynamic(det_path, det_int8, weight_type=QuantType.QInt8)
    print(f"[SIZE] fp32={os.path.getsize(det_path) / 1e6:.0f}MB -> int8={os.path.getsize(det_int8) / 1e6:.0f}MB")

    print("""
[SUMMARY] InsightFace 部署优化:
1. 检测: SCRFD (移动端选 det_500m, 0.5MB) / 识别: w600k_r50 -> 可换 r18 蒸馏版
2. int8 动态量化零校准成本，精度损失 <1%；追求极致用 QAT
3. 端侧进一步: 线程数 (sess.add_session_config), CoreML EP
""")


if __name__ == "__main__":
    main()
