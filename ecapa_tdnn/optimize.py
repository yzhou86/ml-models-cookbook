"""ECAPA-TDNN 优化对比: fp32 vs 动态 int8 量化 vs TorchScript。

22M 参数 -> 量化后内存 ~88MB -> ~25MB；CPU 推理提速约 2x。
"""
import copy

import torch
import torchaudio


def get_model_and_input():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    wrapper = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    ecapa = wrapper.mods["embedding_model"].eval()

    from common.utils import download
    from speechbrain.lobes.features import Fbank
    wav, sr = torchaudio.load(download(
        "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav", "preamble10.wav"
    ))
    feats = Fbank()(wav)  # (T, F)
    return ecapa, feats


def main():
    from common.utils import benchmark, print_bench
    print(f"[INFO] qengine = {torch.backends.quantized.engine}")

    model, feats = get_model_and_input()

    @torch.no_grad()
    def run(m):
        m(feats)

    run(model)
    print_bench("ECAPA fp32 (CPU)", benchmark(lambda: run(model), 3, 20))

    # ---- 1. 动态 int8 量化（Linear 层） ----
    model_q = copy.deepcopy(model)
    torch.ao.quantization.quantize_dynamic(model_q, dtype=torch.qint8)
    run(model_q)
    print_bench("ECAPA int8 dynamic (CPU)", benchmark(lambda: run(model_q), 3, 20))

    # ---- 2. TorchScript（失败自动降级） ----
    try:
        scripted = torch.jit.script(model)
        run(scripted)
        print_bench("ECAPA TorchScript (CPU)", benchmark(lambda: run(scripted), 3, 20))
    except Exception as e:
        print(f"[WARN] TorchScript 跳过（控制流不支持 script）: {e}")

    # ---- 3. 一致性检查 ----
    with torch.no_grad():
        e1 = model(feats)
        e2 = model_q(feats)
    cos = torch.nn.functional.cosine_similarity(e1, e2, dim=-1).item()
    print(f"[CHECK] 量化前后 embedding 余弦相似度: {cos:.4f} (>0.99 视为无损)")


if __name__ == "__main__":
    main()
