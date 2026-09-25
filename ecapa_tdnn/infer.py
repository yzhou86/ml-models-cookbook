"""ECAPA-TDNN (SpeechBrain) 说话人验证推理。

用法:
    python -m ecapa_tdnn.infer                    # 下载示例音频自比较
    python -m ecapa_tdnn.infer --a a.wav --b b.wav
模型: speechbrain/spkrec-ecapa-voxceleb (~22M 参数, 192 维 embedding)
"""

import argparse

import torch


def load_ecapa(device: str = "cpu"):
    """兼容 speechbrain 1.x（inference 模块）与旧版（pretrained 模块）。"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},  # 默认 CPU；SpeechBrain 部分算子在 MPS 覆盖不完整
    )
    return model


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def encode_file(model, path: str) -> torch.Tensor:
    """兼容 SpeechBrain 1.x：显式读取波形后调用 encode_batch。"""
    waveform = model.load_audio(path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return model.encode_batch(waveform).flatten(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None, help="音频1 (16k wav)")
    parser.add_argument("--b", default=None, help="音频2 (16k wav)")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="仅作演示；生产阈值必须在目标数据集上按 FAR/FRR 标定"
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    from common.utils import download

    sample = download("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", "jfk.flac")
    path_a = args.a or sample
    path_b = args.b or sample  # 缺省同一段音频自比较（应≈1.0）

    model = load_ecapa(args.device)

    emb_a = encode_file(model, path_a)  # (1, 192)
    emb_b = encode_file(model, path_b)

    score = cosine(emb_a, emb_b)
    print(f"[INFO] embedding dim: {emb_a.shape[-1]}")
    print(f"[RESULT] cosine similarity = {score:.4f}")
    print(
        f"[RESULT] 判定: {'同一说话人' if score > args.threshold else '不同说话人'} "
        f"(演示阈值 {args.threshold})"
    )


if __name__ == "__main__":
    main()
