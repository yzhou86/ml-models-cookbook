"""ECAPA-TDNN (SpeechBrain) 说话人验证推理。

用法:
    python -m ecapa_tdnn.infer                    # 下载示例音频自比较
    python -m ecapa_tdnn.infer --a a.wav --b b.wav
模型: speechbrain/spkrec-ecapa-voxceleb (~22M 参数, 192 维 embedding)
"""
import argparse

import torch


def load_ecapa():
    """兼容 speechbrain 1.x（inference 模块）与旧版（pretrained 模块）。"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},  # ECAPA 算子在 MPS 覆盖不全，用 CPU（性能足够）
    )
    return model


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None, help="音频1 (16k wav)")
    parser.add_argument("--b", default=None, help="音频2 (16k wav)")
    args = parser.parse_args()

    from common.utils import download
    sample = download(
        "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav", "preamble10.wav"
    )
    path_a = args.a or sample
    path_b = args.b or sample  # 缺省同一段音频自比较（应≈1.0）

    model = load_ecapa()

    emb_a = model.encode_file(path_a)          # (1, 192)
    emb_b = model.encode_file(path_b)

    score = cosine(emb_a, emb_b)
    print(f"[INFO] embedding dim: {emb_a.shape[-1]}")
    print(f"[RESULT] cosine similarity = {score:.4f}")
    print(f"[RESULT] 判定: {'同一说话人' if score > 0.75 else '不同说话人'} (阈值 0.75)")


if __name__ == "__main__":
    main()
