"""Fine-tune Helsinki-NLP OPUS-MT models on parallel bilingual corpus.

Supports bidirectional fine-tuning (zh→en and en→zh).

Usage:
    # Fine-tune zh→en (default)
    python train.py --direction zh2en

    # Fine-tune en→zh
    python train.py --direction en2zh

    # Fine-tune both sequentially
    python train.py --direction both

Data: expects line-aligned train_zh.txt / train_en.txt in --data_dir.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ──────────────────────────────────────────────────────

MODEL_MAP = {
    "zh2en": "Helsinki-NLP/opus-mt-zh-en",
    "en2zh": "Helsinki-NLP/opus-mt-en-zh",
}

OUTPUT_DIRS = {
    "zh2en": "./output_zh2en",
    "en2zh": "./output_en2zh",
}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune OPUS-MT models")
    p.add_argument("--direction", default="zh2en",
                   choices=["zh2en", "en2zh", "both"],
                   help="Translation direction")
    p.add_argument("--data_dir", default="./archive_5",
                   help="Directory with train_en.txt / train_zh.txt")
    p.add_argument("--max_samples", type=int, default=2_000_000,
                   help="Max training samples (per direction)")
    p.add_argument("--val_samples", type=int, default=2000,
                   help="Validation samples")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--max_length", type=int, default=128,
                   help="Max sequence length")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--use_lora", action="store_true", default=True,
                   help="Use LoRA for efficient fine-tuning")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use mixed precision")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data loading ────────────────────────────────────────────────


def load_data(data_dir: str, direction: str, max_samples: int,
              val_samples: int, tokenizer, max_length: int):
    """Load parallel data and return train/val HuggingFace Datasets."""
    en_path = os.path.join(data_dir, "train_en.txt")
    zh_path = os.path.join(data_dir, "train_zh.txt")

    print(f"Loading data from {data_dir} ...")
    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = [ln.strip() for ln in f.readlines()]
    with open(zh_path, "r", encoding="utf-8") as f:
        zh_lines = [ln.strip() for ln in f.readlines()]

    n = min(len(en_lines), len(zh_lines), max_samples)
    # Take first N lines for training, last val_samples for validation
    if direction == "zh2en":
        source_lines = zh_lines[:n]
        target_lines = en_lines[:n]
        val_src = zh_lines[-val_samples:]
        val_tgt = en_lines[-val_samples:]
        tok_src_lang = "zho"
        tok_tgt_lang = "eng"
    else:
        source_lines = en_lines[:n]
        target_lines = zh_lines[:n]
        val_src = en_lines[-val_samples:]
        val_tgt = zh_lines[-val_samples:]
        tok_src_lang = "eng"
        tok_tgt_lang = "zho"

    # Free raw text memory
    del en_lines, zh_lines

    # Set tokenizer language
    tokenizer.source_lang = tok_src_lang
    tokenizer.tgt_lang = tok_tgt_lang
    # Set tokenizer target language for generation
    tokenizer.tgt_lang = tok_tgt_lang

    print(f"  {direction}: {n:,} train pairs, {val_samples} val pairs")

    def preprocess(batch):
        """Tokenize source → target."""
        model_inputs = tokenizer(
            batch["source"], max_length=max_length,
            truncation=True, padding=False,
        )
        labels = tokenizer(
            batch["target"], max_length=max_length,
            truncation=True, padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_raw = Dataset.from_dict({
        "source": source_lines,
        "target": target_lines,
    })
    val_raw = Dataset.from_dict({
        "source": val_src,
        "target": val_tgt,
    })

    train_ds = train_raw.map(preprocess, batched=True,
                             remove_columns=train_raw.column_names,
                             desc="Tokenizing train")
    val_ds = val_raw.map(preprocess, batched=True,
                         remove_columns=val_raw.column_names,
                         desc="Tokenizing val")

    return train_ds, val_ds


# ── Main ────────────────────────────────────────────────────────


def fine_tune(direction: str, args):
    """Run fine-tuning for a single direction."""
    set_seed(args.seed)

    model_name = MODEL_MAP[direction]
    output_dir = OUTPUT_DIRS[direction]

    print(f"\n{'='*60}")
    print(f"Fine-tuning {direction}: {model_name}")
    print(f"{'='*60}")

    # Load model & tokenizer
    print(f"[1/4] Loading {model_name} ...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "fc1", "fc2"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    print(f"[2/4] Loading data ...")
    train_ds, val_ds = load_data(
        args.data_dir, direction, args.max_samples,
        args.val_samples, tokenizer, args.max_length,
    )

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=500,
        num_train_epochs=args.epochs,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=200,
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Train
    print(f"[3/4] Training ...")
    trainer.train()

    # Save final model
    print(f"[4/4] Saving to {output_dir} ...")
    # Merge LoRA if used, then save
    if args.use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Done: {output_dir}/")


def main():
    args = parse_args()

    if args.direction == "both":
        fine_tune("zh2en", args)
        fine_tune("en2zh", args)
    else:
        fine_tune(args.direction, args)


if __name__ == "__main__":
    main()
