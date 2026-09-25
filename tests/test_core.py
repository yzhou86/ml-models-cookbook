"""不下载模型的快速回归测试；用于提前发现张量形状和数学实现错误。"""

import numpy as np
import torch

from insightface_arcface.infer import ARCFACE_DST, align_face, nms
from insightface_arcface.train import ArcMarginProduct
from silero_vad.infer import get_timestamps
from silero_vad.train_custom_vad import TinyVAD, logmel
from whisper_asr.train_finetune import make_collator


def test_vad_feature_and_model_time_dimensions_match():
    wav = np.zeros(32000, dtype=np.float32)
    features = logmel(wav)
    model = TinyVAD()
    logits = model(torch.from_numpy(features)[None])
    assert logits.shape == (1, len(features))


def test_vad_timestamp_hysteresis_and_final_frame():
    probs = [(i * 0.032, p) for i, p in enumerate([0.1, 0.7, 0.8, 0.6, 0.2, 0.1, 0.1])]
    segments = get_timestamps(
        probs, threshold=0.5, end_threshold=0.35, min_speech_duration=0.05, min_silence_duration=0.064
    )
    assert segments == [(0.032, 0.128)]


def test_arcface_alignment_is_identity_for_template_points():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)
    aligned = align_face(image, ARCFACE_DST.copy())
    assert np.mean(np.abs(aligned.astype(np.int16) - image.astype(np.int16))) < 1.0


def test_nms_suppresses_overlapping_box():
    detections = np.array(
        [
            [0, 0, 10, 10, 0.9, *([0] * 10)],
            [1, 1, 11, 11, 0.8, *([0] * 10)],
            [30, 30, 40, 40, 0.7, *([0] * 10)],
        ],
        dtype=np.float32,
    )
    assert nms(detections, 0.4) == [0, 2]


def test_arc_margin_is_finite_and_backpropagates():
    layer = ArcMarginProduct(8, 3)
    embeddings = torch.randn(4, 8, requires_grad=True)
    logits = layer(embeddings, torch.tensor([0, 1, 2, 1]))
    logits.sum().backward()
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embeddings.grad).all()


def test_whisper_collator_accepts_list_of_examples():
    class Batch(dict):
        __getattr__ = dict.__getitem__

    class FeatureExtractor:
        @staticmethod
        def pad(items, return_tensors):
            return Batch(
                input_features=torch.tensor([item["input_features"] for item in items]),
                attention_mask=torch.tensor([item["attention_mask"] for item in items]),
            )

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def pad(items, return_tensors):
            width = max(len(item["input_ids"]) for item in items)
            ids, masks = [], []
            for item in items:
                values = item["input_ids"]
                ids.append(values + [0] * (width - len(values)))
                masks.append([1] * len(values) + [0] * (width - len(values)))
            return Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))

    class Processor:
        feature_extractor = FeatureExtractor()
        tokenizer = Tokenizer()

    collate = make_collator(Processor(), decoder_start_token_id=1)
    batch = collate(
        [
            {"input_features": [[1.0, 2.0]], "attention_mask": [1], "labels": [1, 4, 5]},
            {"input_features": [[3.0, 4.0]], "attention_mask": [1], "labels": [1, 6]},
        ]
    )
    assert batch["input_features"].shape == (2, 1, 2)
    assert batch["labels"].tolist() == [[4, 5], [6, -100]]
