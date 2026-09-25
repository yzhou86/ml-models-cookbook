"""Whisper 小数据集微调（LoRA 思路的冻结编码器版本 + 完整微调开关）。

演示如何在 24GB 的 MacBook Air M5 上微调 whisper-tiny：
- 数据集: facebook/minds14（en 子集，小而快），可替换为自己的数据
- 技巧: 冻结 encoder 只训 decoder（参数减半）；支持 AMP 半精度

用法:
    python -m whisper_asr.train_finetune                          # 冻结 encoder
    python -m whisper_asr.train_finetune --train_encoder          # 全参数微调
"""
import argparse
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from common.utils import get_device

MODEL_NAME = "openai/whisper-tiny"


def prepare_dataset(processor, dataset):
    """重采样到 16k + 提取 log-mel 特征 + 编码标签。"""

    def map_fn(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    return dataset.map(map_fn, remove_columns=dataset.column_names)


def make_collator(processor):
    def collate_fn(features):
        input_features = [{"input_features": f} for f in features["input_features"]]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = processor.tokenizer.pad(
            [{"input_ids": f} for f in features["labels"]], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_encoder", action="store_true", help="同时训练 encoder（默认冻结）")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--output", default="whisper_finetuned")
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] device={device}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    if not args.train_encoder:
        for p in model.model.encoder.parameters():  # 小内存微调核心技巧: 冻结 encoder
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] trainable params: {trainable / 1e6:.1f}M")

    # 小数据集，跑通全流程；实际项目替换成自己的 ASR 数据集
    from datasets import Audio
    ds = load_dataset("facebook/minds14", "en-US", split="train[:200]")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = prepare_dataset(processor, ds)
    train_ds, eval_ds = ds.select(range(160)), ds.select(range(160, 200))

    try:
        wer = evaluate.load("wer")
    except Exception:
        wer = None

    def compute_metrics(pred):
        if wer is None:
            return {}
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_texts = processor.batch_decode(pred_ids)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        refs = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer.compute(predictions=pred_texts, references=refs)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        fp16=False,  # MPS 不支持 fp16 GradScaler；AMP 由 accelerate 处理
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        use_cpu=(device.type == "cpu"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.generation_config.language = "en"
    trainer.save_model(os.path.join(args.output, "final"))
    processor.save_pretrained(os.path.join(args.output, "final"))
    print(f"[DONE] saved to {args.output}/final")


if __name__ == "__main__":
    main()
