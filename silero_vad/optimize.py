"""Silero VAD 优化对比: JITScript vs ONNX Runtime。

Silero v5 ONNX 接口: inputs = (input [B,512], state [2,B,128], sr int64)
state 是有状态的 LSTM 上下文，每次前向传入并接收更新。
"""
import os
import time

import numpy as np
import torch
from torch.hub import download_url_to_file

CHUNK = 512


def bench(fn, n_warmup=3, n_runs=50):
    for _ in range(n_warmup):
        fn()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - start) / n_runs * 1000


def load_wav():
    from common.utils import download
    import soundfile as sf
    wav_path = download(
        "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav", "preamble10.wav"
    )
    return sf.read(wav_path, dtype="float32")[0]


def main():
    from common.utils import print_bench
    wav = load_wav()
    n_chunks = (len(wav) + CHUNK - 1) // CHUNK

    # ---- 1. torch JITScript ----
    model_jit, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)

    def run_jit():
        model_jit.reset_states()
        with torch.no_grad():
            for i in range(n_chunks):
                chunk = wav[i * CHUNK: (i + 1) * CHUNK]
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
                model_jit(torch.from_numpy(chunk), 16000)

    run_jit()
    print_bench("Silero JITScript (CPU)", bench(run_jit))

    # ---- 2. ONNX Runtime ----
    try:
        import onnxruntime as ort
        onnx_path = "silero_vad.onnx"
        if not os.path.exists(onnx_path):
            download_url_to_file(
                "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
                onnx_path,
            )
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        print(f"[INFO] ONNX inputs: {names}")

        def run_onnx():
            state = np.zeros((2, 1, 128), dtype=np.float32)
            for i in range(n_chunks):
                chunk = wav[i * CHUNK: (i + 1) * CHUNK]
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))[None, :].astype(np.float32)
                feeds = {names[0]: chunk, names[1]: state, names[2]: np.array(16000, dtype=np.int64)}
                _, state = sess.run(None, feeds)

        run_onnx()
        print_bench("Silero ONNX Runtime (CPU)", bench(run_onnx))
    except Exception as e:
        print(f"[WARN] ONNX 分支跳过: {e}")

    print("""
[SUMMARY] Silero 优化实践:
1. 部署首选 ONNX Runtime（官方提供 .onnx，CPU 上通常快 20~50%）
2. 模型仅 ~2MB，量化收益有限，重点在滑窗 IO 优化与批处理
3. 服务场景可多路音频拼 batch 推理，吞吐提升接近线性
""")


if __name__ == "__main__":
    main()
