"""Pre-tokenize all train/valid data into .pt shards for fast training.

One-time cost: ~2 hours CPU, ~20 GB disk per 4M sample direction.
Training speed: 30-40 it/s (was 8-10 it/s with on-the-fly tokenize).
"""

import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from train.config import Config
from train.bert_embedding import BERTEmbedding

SHARD_SIZE = 500_000  # samples per shard
BATCH_SIZE = 256      # tokenize batch size (HuggingFace internal batch)


def _read_lines(path, offsets, start, count):
    """Read count lines from path starting at line `start` using pre-built offsets."""
    end = min(start + count, len(offsets))
    with open(path, "r", encoding="utf-8") as f:
        for i in range(start, end):
            f.seek(offsets[i])
            yield f.readline().strip()


def tokenize_pair_chunk(args):
    """Tokenize a contiguous chunk of line pairs → dict of tensors."""
    chunk_idx, src_path, tgt_path, tag, start, count, max_len, tokenizer, \
        src_offsets, tgt_offsets = args

    src_lines = list(_read_lines(src_path, src_offsets, start, count))
    tgt_lines = list(_read_lines(tgt_path, tgt_offsets, start, count))

    src_texts = [f"{tag} {s}" for s in src_lines]
    tgt_texts = tgt_lines

    all_src_ids = []
    all_src_mask = []
    all_tgt_ids = []
    all_dec_input = []
    all_tgt_pad = []

    sos_id = tokenizer.cls_token_id or 101
    pad_id = tokenizer.pad_token_id or 0

    for i in range(0, len(src_texts), BATCH_SIZE):
        b_src = src_texts[i:i + BATCH_SIZE]
        b_tgt = tgt_texts[i:i + BATCH_SIZE]

        src_enc = tokenizer(b_src, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
        tgt_enc = tokenizer(b_tgt, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")

        src_ids = src_enc["input_ids"].to(torch.int32)
        src_mask = src_enc["attention_mask"].bool()
        tgt_ids = tgt_enc["input_ids"].to(torch.int32)

        sos = torch.full((tgt_ids.size(0), 1), sos_id, dtype=torch.int32)
        dec_input = torch.cat([sos, tgt_ids[:, :-1]], dim=1)
        tgt_pad = (tgt_ids == pad_id)

        all_src_ids.append(src_ids)
        all_src_mask.append(src_mask)
        all_tgt_ids.append(tgt_ids)
        all_dec_input.append(dec_input)
        all_tgt_pad.append(tgt_pad)

    return {
        "src_ids": torch.cat(all_src_ids, dim=0),
        "src_mask": torch.cat(all_src_mask, dim=0),
        "tgt_ids": torch.cat(all_tgt_ids, dim=0),
        "decoder_input": torch.cat(all_dec_input, dim=0),
        "tgt_pad_mask": torch.cat(all_tgt_pad, dim=0),
    }


def tokenize_split(config, split: str, out_dir: str):
    """Tokenize a full split (train/valid) and save as shards (sequential)."""
    from train.dataset import _build_line_index

    tokenizer = BERTEmbedding(
        config.bert_model_name, config.d_model,
        local_path=config.bert_local_path
    ).tokenizer
    max_len = config.max_len

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for src_lang, tgt_lang in config.lang_pairs:
        tag = config.lang_tags[tgt_lang]
        src_path = os.path.join(config.data_dir, f"{split}.{src_lang}")
        tgt_path = os.path.join(config.data_dir, f"{split}.{tgt_lang}")

        # Build offsets ONCE per file
        print(f"  Indexing {src_path} ...")
        src_offsets = _build_line_index(src_path)
        print(f"  Indexing {tgt_path} ...")
        tgt_offsets = _build_line_index(tgt_path)

        n = min(len(src_offsets), len(tgt_offsets))
        cap = config.max_train_samples if split == "train" else n
        n = min(n, cap)

        print(f"  [{split}] {src_lang}→{tgt_lang}: {n:,} pairs → shards of {SHARD_SIZE:,}")

        n_chunks = (n + SHARD_SIZE - 1) // SHARD_SIZE
        bar = tqdm(total=n_chunks, desc=f"  Tokenizing {split}", unit="shard")

        for start in range(0, n, SHARD_SIZE):
            chunk_count = min(SHARD_SIZE, n - start)
            chunk_idx = start // SHARD_SIZE

            args = (chunk_idx, src_path, tgt_path, tag,
                    start, chunk_count, max_len, tokenizer,
                    src_offsets, tgt_offsets)
            data = tokenize_pair_chunk(args)

            shard_name = f"{split}.{src_lang}.{tgt_lang}.{chunk_idx:04d}.pt"
            torch.save(data, out_path / shard_name)
            n_samples = data["src_ids"].size(0)
            total_samples += n_samples
            bar.update(1)
            bar.set_postfix(samples=f"{total_samples:,}")

        bar.close()

    # Save a manifest
    manifest = {"split": split, "total_samples": total_samples,
                "shard_size": SHARD_SIZE, "max_len": max_len}
    torch.save(manifest, out_path / f"{split}_manifest.pt")

    size_gb = sum(p.stat().st_size for p in out_path.glob("*.pt")) / (1024 ** 3)
    print(f"  [{split}] Done: {total_samples:,} samples, {size_gb:.1f} GB\n")
    return total_samples


def main():
    config = Config()

    out_dir = os.path.join(config.data_dir, "tokenized")
    print(f"Pre-tokenizing → {out_dir}/")
    t0 = time.time()

    train_n = tokenize_split(config, "train", out_dir)
    valid_n = tokenize_split(config, "valid", out_dir)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone: {train_n + valid_n:,} samples in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
