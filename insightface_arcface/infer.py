"""InsightFace (buffalo_l: SCRFD 检测 + ArcFace w600k_r50 识别) 人脸验证推理。

不依赖 insightface pip 包（其 C 扩展在部分 mac 上编译困难），
直接用 onnxruntime 加载官方 ONNX 模型，并在本文件内完整实现
SCRFD anchor 解码 + NMS + 5 点对齐（面试可直接讲原理）。
首次运行自动下载 buffalo_l (~280MB)。

用法:
    python -m insightface_arcface.infer                 # 示例两张人脸对比
    python -m insightface_arcface.infer --a 1.jpg --b 2.jpg
"""

import argparse
import os
import zipfile

import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
CACHE = os.path.expanduser("~/.cache/insightface_models")
STRIDES = (8, 16, 32)
DET_SIZE = (640, 640)
# ArcFace 标准 5 点模板 (112x112)
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def get_model_root() -> str:
    """兼容 release zip 的扁平结构和旧版 buffalo_l 子目录结构。"""
    nested = os.path.join(CACHE, "buffalo_l")
    required = ("det_10g.onnx", "w600k_r50.onnx")
    if all(os.path.isfile(os.path.join(nested, name)) for name in required):
        return nested
    if all(os.path.isfile(os.path.join(CACHE, name)) for name in required):
        return CACHE
    return nested


def ensure_models(threads: int = 0) -> dict:
    """下载/解压 buffalo_l，只加载检测与识别两个必需 ONNX 模型。"""
    root = get_model_root()
    required = ("det_10g.onnx", "w600k_r50.onnx")
    if not all(os.path.isfile(os.path.join(root, name)) for name in required):
        os.makedirs(CACHE, exist_ok=True)
        zip_path = os.path.join(CACHE, "buffalo_l.zip")
        if not os.path.exists(zip_path):
            import urllib.request

            print(f"[DOWNLOAD] {MODEL_URL} (~280MB, 首次运行)")
            urllib.request.urlretrieve(MODEL_URL, zip_path)
        root = os.path.join(CACHE, "buffalo_l")
        os.makedirs(root, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            cache_real = os.path.realpath(root)
            for member in zf.infolist():
                destination = os.path.realpath(os.path.join(root, member.filename))
                if os.path.commonpath([cache_real, destination]) != cache_real:
                    raise RuntimeError(f"压缩包包含不安全路径: {member.filename}")
            zf.extractall(root)
    sessions = {}
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    for name in ("det_10g", "w600k_r50"):
        path = os.path.join(root, f"{name}.onnx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模型不完整，缺少: {path}")
        sessions[name] = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    return sessions


def nms(dets: np.ndarray, thresh: float = 0.4) -> list:
    """dets: (N, 15) [x1,y1,x2,y2,score, kps x10]。"""
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = dets[:, 4].argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


def detector_blob(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """SCRFD letterbox 预处理，返回 NCHW blob 与缩放率。"""
    h, w = img_bgr.shape[:2]
    ratio = min(DET_SIZE[0] / w, DET_SIZE[1] / h)
    resized = cv2.resize(img_bgr, (round(w * ratio), round(h * ratio)))
    canvas = np.zeros((DET_SIZE[1], DET_SIZE[0], 3), dtype=np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    # 官方 SCRFD blobFromImage 使用 swapRB=True。
    blob = ((canvas[:, :, ::-1] - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
    return blob, ratio


def detect_faces(det_sess, img_bgr: np.ndarray, det_thresh: float = 0.5) -> np.ndarray:
    """SCRFD 解码。输出顺序: [3 组 scores, 3 组 bbox, 3 组 kps]。

    bbox 是相对 anchor 中心的 4 距离（乘 stride），kps 同理 -> 标准 anchor-based 解码。
    返回 (N, 15): [x1,y1,x2,y2,score, 10 个关键点坐标]。
    """
    blob, ratio = detector_blob(img_bgr)

    outputs = det_sess.run(None, {det_sess.get_inputs()[0].name: blob})
    n = len(STRIDES)
    all_scores, all_boxes, all_kps = [], [], []
    for i, stride in enumerate(STRIDES):
        # buffalo_l release 的输出可能是 (A,C) 或 (1,A,C)，统一展平批维。
        scores = outputs[i].reshape(-1, 1)
        bbox_d = outputs[i + n].reshape(-1, 4)
        kps_d = outputs[i + 2 * n].reshape(-1, 10)

        H, W = DET_SIZE[1] // stride, DET_SIZE[0] // stride
        centers = np.stack(np.mgrid[:H, :W][::-1], -1).astype(np.float32)  # (H, W, 2)
        centers = (centers * stride).reshape(-1, 2)
        centers = np.repeat(centers, 2, axis=0)  # 每格 2 个 anchor -> (A, 2)

        bbox = np.concatenate([centers - bbox_d[:, :2] * stride, centers + bbox_d[:, 2:] * stride], axis=1)
        kps_x = centers[:, :1] + kps_d[:, 0::2] * stride
        kps_y = centers[:, 1:] + kps_d[:, 1::2] * stride
        kps = np.stack([kps_x, kps_y], -1).reshape(-1, 10)

        all_scores.append(scores[:, 0])
        all_boxes.append(bbox)
        all_kps.append(kps)

    scores = np.concatenate(all_scores)
    boxes = np.concatenate(all_boxes) / ratio
    kps = np.concatenate(all_kps) / ratio
    keep = scores > det_thresh
    dets = np.concatenate([boxes, scores[:, None], kps], axis=1)[keep]
    if len(dets) == 0:
        return dets
    return dets[nms(dets)]


def align_face(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """5 点相似变换对齐到 112x112（ArcFace 标准输入）。"""
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("无法根据 5 点关键点估计人脸相似变换")
    return cv2.warpAffine(img_bgr, matrix, (112, 112), borderValue=0)


def arcface_blob(aligned_bgr: np.ndarray) -> np.ndarray:
    """ArcFace 标准 RGB、[-1, 1]、NCHW 预处理。"""
    blob = aligned_bgr[:, :, ::-1].astype(np.float32)
    return ((blob - 127.5) / 127.5).transpose(2, 0, 1)[None]


def arcface_embed(rec_sess, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    aligned = align_face(img_bgr, kps)
    blob = arcface_blob(aligned)
    emb = rec_sess.run(None, {rec_sess.get_inputs()[0].name: blob})[0][0]
    return emb / np.linalg.norm(emb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None)
    parser.add_argument("--b", default=None)
    parser.add_argument("--thresh", type=float, default=0.35, help="cosine 阈值，insightface 官方推荐 ~0.35")
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op 线程数；0 表示默认")
    args = parser.parse_args()

    from common.utils import download

    path_a = args.a or download(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "lena.jpg",
    )
    path_b = args.b or download(
        "https://raw.githubusercontent.com/deepinsight/insightface/master/"
        "python-package/insightface/data/images/t1.jpg",
        "faces_t1.jpg",
    )

    sessions = ensure_models(args.threads)
    print(f"[INFO] loaded ONNX: {sorted(sessions)}")
    det, rec = sessions["det_10g"], sessions["w600k_r50"]

    embeddings = []
    for p in [path_a, path_b]:
        img = cv2.imread(p)
        if img is None:
            print(f"[ERROR] 无法读取 {p}")
            return
        dets = detect_faces(det, img, args.det_thresh)
        if len(dets) == 0:
            print(f"[WARN] {p} 未检测到人脸（换 --a/--b 传自己的人脸照片）")
            return
        best = dets[np.argmax(dets[:, 4])]
        embeddings.append(arcface_embed(rec, img, best[5:15]))
        print(f"[INFO] {p}: face bbox={np.round(best[:4]).astype(int).tolist()}, conf={best[4]:.2f}")

    cos = float(np.dot(*embeddings))
    print(f"[RESULT] cosine similarity = {cos:.4f}")
    print(f"[RESULT] 判定: {'同一人' if cos > args.thresh else '不同人'} (阈值 {args.thresh})")


if __name__ == "__main__":
    main()
