"""Whisper 小数据集微调（LoRA 思路的冻结编码器版本 + 完整微调开关）。

演示如何在 24GB 的 MacBook Air M5 上微调 whisper-tiny：
- 数据集: PolyAI/minds14（en-US 子集，小而快），可替换为自己的数据
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
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from common.utils import get_device, seed_everything
from whisper_asr.infer import resolve_model_name

MODEL_NAME = "openai/whisper-tiny"


def prepare_dataset(processor, dataset):
    """重采样到 16k + 提取 log-mel 特征 + 编码标签。"""

    def map_fn(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        batch["input_features"] = inputs.input_features[0]
        batch["attention_mask"] = inputs.attention_mask[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    return dataset.map(map_fn, remove_columns=dataset.column_names)


def make_collator(processor, decoder_start_token_id: int | None = None):
    def collate_fn(features):
        # Trainer/DataLoader 传入的是 List[Dict]，不是 Dict[List]。
        input_features = [
            {
                "input_features": f["input_features"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if decoder_start_token_id is not None and torch.all(labels[:, 0] == decoder_start_token_id):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_encoder", action="store_true", help="同时训练 encoder（默认冻结）")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--eval-samples", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2, help="M5 24GB 建议 tiny=2~4、base=1~2、small=1")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="whisper_finetuned")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"[INFO] device={device}")

    model_name = resolve_model_name(args.model)
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if not args.train_encoder:
        for p in model.model.encoder.parameters():  # 小内存微调核心技巧: 冻结 encoder
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] trainable params: {trainable / 1e6:.1f}M")

    # 小数据集，跑通全流程；实际项目替换成自己的 ASR 数据集
    from datasets import Audio

    requested = args.train_samples + args.eval_samples
    ds = load_dataset("PolyAI/minds14", "en-US", split=f"train[:{requested}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = prepare_dataset(processor, ds)
    train_end = min(args.train_samples, len(ds))
    eval_end = min(train_end + args.eval_samples, len(ds))
    train_ds = ds.select(range(train_end))
    eval_ds = ds.select(range(train_end, eval_end))
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError("训练集或验证集为空，请增大 --train-samples/--eval-samples")

    try:
        wer = evaluate.load("wer")
    except Exception:
        wer = None

    def compute_metrics(pred):
        if wer is None:
            return {}
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        if pred_ids.ndim == 3:  # 兼容未启用 generate 时返回 logits 的情况
            pred_ids = np.argmax(pred_ids, axis=-1)
        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        refs = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer.compute(predictions=pred_texts, references=refs)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,  # Trainer 的 CUDA fp16/GradScaler 路径不适用于 MPS
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=128,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],
        use_cpu=(device.type == "cpu"),
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.generation_config.language = "en"
    trainer.save_model(os.path.join(args.output, "final"))
    processor.save_pretrained(os.path.join(args.output, "final"))
    print(f"[DONE] saved to {args.output}/final")


if __name__ == "__main__":
    main()
