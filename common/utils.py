"""公共工具：设备检测、计时、示例文件下载。"""
import functools
import os
import time
import urllib.request

import torch

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)


def get_device():
    """训练/浮点推理设备：mps(Apple Silicon) > cuda > cpu。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_qengine():
    """ARM 芯片量化后端：qnnpack（fbgemm 仅 x86）。"""
    return torch.backends.quantized.engine


def timer(func):
    """计时装饰器，打印耗时。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.manual_seed(0)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed * 1000:.1f} ms")
        return result

    return wrapper


def download(url: str, filename: str) -> str:
    """下载示例文件到 samples/，返回本地路径（已存在则直接复用）。"""
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(path):
        print(f"[DOWNLOAD] {url}")
        urllib.request.urlretrieve(url, path)
    return path


def benchmark(fn, n_warmup: int = 3, n_runs: int = 10):
    """基准测试：预热 n_warmup 次后正式计时 n_runs 次，返回平均毫秒。"""
    for _ in range(n_warmup):
        fn()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - start) / n_runs * 1000


def print_bench(title: str, ms: float):
    print(f"[BENCH] {title:<30s} {ms:8.2f} ms/run")
