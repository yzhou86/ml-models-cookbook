"""InsightFace ONNX 静态 PTQ：真实人脸校准、延迟、大小与输出一致性。"""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort


def bench(sess, feed, n_warmup=3, n_runs=20):
    import time

    for _ in range(n_warmup):
        sess.run(None, feed)
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feed)
    return (time.perf_counter() - start) / n_runs * 1000


def quantize_qdq(source: str, target: str, arrays: list[np.ndarray], rebuild: bool) -> None:
    if os.path.exists(target) and not rebuild:
        return
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    input_name = ort.InferenceSession(source, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter([{input_name: array} for array in arrays])

        def get_next(self):
            return next(self.items, None)

    quantize_static(
        source,
        target,
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # buffalo_l 是 opset 11；per-channel QDQ 的 axis 属性需更高 opset，故使用 per-tensor。
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"WeightSymmetric": True},
    )


def make_session(path: str, threads: int):
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-detector", action="store_true", help="只量化 ArcFace，节省校准时间")
    args = parser.parse_args()

    from common.utils import download, print_bench
    from insightface_arcface.infer import (
        align_face,
        arcface_blob,
        detect_faces,
        detector_blob,
        ensure_models,
        get_model_root,
    )

    sessions = ensure_models(args.threads)
    det_fp32, rec_fp32 = sessions["det_10g"], sessions["w600k_r50"]
    root = get_model_root()

    paths = [
        download(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            "lena.jpg",
        ),
        download(
            "https://raw.githubusercontent.com/deepinsight/insightface/master/"
            "python-package/insightface/data/images/t1.jpg",
            "faces_t1.jpg",
        ),
    ]
    det_inputs, rec_inputs, images = [], [], []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"无法读取校准图: {path}")
        images.append(image)
        det_inputs.append(detector_blob(image)[0])
        dets = detect_faces(det_fp32, image)
        if len(dets) == 0:
            raise RuntimeError(f"校准图未检测到人脸: {path}")
        best = dets[np.argmax(dets[:, 4])]
        rec_inputs.append(arcface_blob(align_face(image, best[5:15])))

    rec_path = os.path.join(root, "w600k_r50.onnx")
    rec_int8 = os.path.join(root, "w600k_r50_qdq_int8.onnx")
    print(f"[QUANT] ArcFace QDQ PTQ, calibration faces={len(rec_inputs)}")
    quantize_qdq(rec_path, rec_int8, rec_inputs, args.rebuild)
    rec_q = make_session(rec_int8, args.threads)
    feed_fp32 = {rec_fp32.get_inputs()[0].name: rec_inputs[0]}
    feed_q = {rec_q.get_inputs()[0].name: rec_inputs[0]}
    print_bench("ArcFace fp32 CPU", bench(rec_fp32, feed_fp32, n_runs=args.runs))
    print_bench("ArcFace int8 QDQ CPU", bench(rec_q, feed_q, n_runs=args.runs))
    e1 = rec_fp32.run(None, feed_fp32)[0][0]
    e2 = rec_q.run(None, feed_q)[0][0]
    cosine = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    print(f"[CHECK] fp32/int8 embedding cosine={cosine:.5f}")
    print(
        f"[SIZE] ArcFace fp32={os.path.getsize(rec_path) / 1e6:.1f}MB -> "
        f"int8={os.path.getsize(rec_int8) / 1e6:.1f}MB"
    )

    if not args.skip_detector:
        det_path = os.path.join(root, "det_10g.onnx")
        det_int8 = os.path.join(root, "det_10g_qdq_int8.onnx")
        print(f"[QUANT] SCRFD QDQ PTQ, calibration images={len(det_inputs)}")
        quantize_qdq(det_path, det_int8, det_inputs, args.rebuild)
        det_q = make_session(det_int8, args.threads)
        feed_det = {det_fp32.get_inputs()[0].name: det_inputs[0]}
        feed_det_q = {det_q.get_inputs()[0].name: det_inputs[0]}
        print_bench("SCRFD fp32 CPU", bench(det_fp32, feed_det, n_runs=args.runs))
        print_bench("SCRFD int8 QDQ CPU", bench(det_q, feed_det_q, n_runs=args.runs))
        before, after = detect_faces(det_fp32, images[0]), detect_faces(det_q, images[0])
        print(f"[CHECK] detector faces: fp32={len(before)}, int8={len(after)}")
        print(
            f"[SIZE] SCRFD fp32={os.path.getsize(det_path) / 1e6:.1f}MB -> "
            f"int8={os.path.getsize(det_int8) / 1e6:.1f}MB"
        )

    print("[NOTE] 两张示例图只验证流程；正式模型必须用代表性校准集并报告验证集 ROC/TAR@FAR。")


if __name__ == "__main__":
    main()
