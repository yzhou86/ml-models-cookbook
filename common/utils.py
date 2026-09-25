"""公共工具：设备、可复现性、音频读取、下载与基准测试。"""

import functools
import hashlib
import io
import math
import os
import random
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def get_device(requested: str = "auto") -> torch.device:
    """解析运行设备；Mac 默认优先 MPS，也允许显式选择 cpu/cuda/mps。"""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("请求了 MPS，但当前 PyTorch/Mac 环境没有可用的 MPS 后端")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前环境没有可用的 CUDA 后端")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_qengine():
    """选择当前 PyTorch 支持的量化后端；Apple Silicon 优先 qnnpack。"""
    supported = torch.backends.quantized.supported_engines
    preferred = "qnnpack" if "qnnpack" in supported else torch.backends.quantized.engine
    if preferred:
        torch.backends.quantized.engine = preferred
    return torch.backends.quantized.engine


def seed_everything(seed: int = 42) -> None:
    """设置教学实验常用随机源，尽量保证结果可复现。"""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)


def timer(func):
    """计时装饰器，打印耗时。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed * 1000:.1f} ms")
        return result

    return wrapper


def download(url: str, filename: str, sha256: str | None = None) -> str:
    """下载到 samples/；临时文件完成后再替换，避免中断后复用残缺文件。"""
    path = SAMPLE_DIR / filename

    def valid() -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if sha256 is None:
            return True
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest.lower() == sha256.lower()

    if not valid():
        print(f"[DOWNLOAD] {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "ml-models-cookbook/0.2"})
        fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", dir=SAMPLE_DIR)
        try:
            with os.fdopen(fd, "wb") as dst, urllib.request.urlopen(request, timeout=60) as src:
                shutil.copyfileobj(src, dst)
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        if not valid():
            path.unlink(missing_ok=True)
            raise RuntimeError(f"下载文件校验失败: {filename}")
    return str(path)


def load_audio_mono(path: str, target_sr: int = 16000):
    """用 soundfile 读取单声道 float32，并在 CPU 上可靠重采样。"""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // divisor, sr // divisor).astype(np.float32)
        sr = target_sr
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def synchronize(device: torch.device | str | None = None) -> None:
    """异步设备计时前同步；CPU 无需处理。"""
    if device is None:
        return
    device_type = torch.device(device).type
    if device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(fn, n_warmup: int = 3, n_runs: int = 10, device=None) -> float:
    """预热后测量平均延迟；自动关闭梯度并同步 MPS/CUDA。"""
    if n_runs < 1:
        raise ValueError("n_runs 必须大于 0")
    with torch.inference_mode():
        for _ in range(n_warmup):
            fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(n_runs):
            fn()
        synchronize(device)
    return (time.perf_counter() - start) / n_runs * 1000


def model_size_mb(model: torch.nn.Module) -> float:
    """序列化 state_dict 后统计大小，能覆盖动态量化的 packed 权重。"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


def print_bench(title: str, ms: float):
    print(f"[BENCH] {title:<30s} {ms:8.2f} ms/run")
