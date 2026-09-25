"""ECAPA-TDNN 优化对比: fp32 vs 动态 int8 量化 vs TorchScript。

动态量化只覆盖 Linear；ECAPA 以 Conv1d 为主，因此脚本重点验证收益是否真实存在。
"""

import argparse
import copy

import torch
import torch.nn as nn


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
    from common.utils import download, load_audio_mono

    wav, _ = load_audio_mono(
        download("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", "jfk.flac"),
        target_sr=16000,
    )
    waveform = torch.from_numpy(wav)[None]
    lengths = torch.ones(1)
    with torch.inference_mode():
        feats = wrapper.mods["compute_features"](waveform)
        feats = wrapper.mods["mean_var_norm"](feats, lengths)
    ecapa = wrapper.mods["embedding_model"].eval()
    return ecapa, feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    from common.utils import benchmark, get_qengine, model_size_mb, print_bench

    print(f"[INFO] qengine = {get_qengine()}")

    model, feats = get_model_and_input()

    @torch.no_grad()
    def run(m):
        m(feats)

    run(model)
    print_bench("ECAPA fp32 (CPU)", benchmark(lambda: run(model), 3, args.runs))

    # ---- 1. 动态 int8 量化（Linear 层） ----
    model_q = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    run(model_q)
    print_bench("ECAPA int8 dynamic (CPU)", benchmark(lambda: run(model_q), 3, args.runs))
    print(f"[SIZE] fp32={model_size_mb(model):.1f}MB -> int8={model_size_mb(model_q):.1f}MB")

    # ---- 2. TorchScript（失败自动降级） ----
    try:
        scripted = torch.jit.script(model)
        run(scripted)
        print_bench("ECAPA TorchScript (CPU)", benchmark(lambda: run(scripted), 3, args.runs))
    except Exception as e:
        print(f"[WARN] TorchScript 跳过（控制流不支持 script）: {e}")

    # ---- 3. 一致性检查 ----
    with torch.no_grad():
        e1 = model(feats)
        e2 = model_q(feats)
    cos = torch.nn.functional.cosine_similarity(e1.flatten(1), e2.flatten(1), dim=-1).item()
    print(f"[CHECK] 量化前后 embedding 余弦相似度: {cos:.4f}（仍需在验证对上检查 EER）")


if __name__ == "__main__":
    main()
