"""Bidirectional data loading for machine translation (memory-efficient).

Each (src, tgt) line pair produces two samples by swapping source/target
and prepending a target-language tag:

    "<2zh> Hello world"  →  "你好世界"
    "<2en> 你好世界"       →  "Hello world"

Lines are read on-the-fly via byte-offset index — file contents are never
fully loaded into memory.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

from train.bert_embedding import BERTEmbedding


def _build_line_index(path):
    """Return a list of byte offsets for every line in a text file."""
    offsets = []
    with open(path, "rb") as f:
        offsets.append(0)
        while f.readline():
            offsets.append(f.tell())
    offsets.pop()  # remove trailing offset beyond last line
    return offsets


class TranslationDataset(Dataset):
    """Random-access dataset indexed by (line_no, direction)."""

    def __init__(self, config, split: str = "train", skip_samples: int = 0):
        self.config = config
        self.max_len = config.max_len
        self.data_dir = config.data_dir

        self.tokenizer = BERTEmbedding(
            config.bert_model_name, config.d_model,
            local_path=config.bert_local_path
        ).tokenizer

        self.samples = []  # (line_no, src_lang, tgt_lang, tag)
        self.offsets = {}  # lang → file path
        self.file_paths = {}  # lang → file path for reading

        for src_lang, tgt_lang in config.lang_pairs:
            tag = config.lang_tags[tgt_lang]

            src_path = os.path.join(config.data_dir, f"{split}.{src_lang}")
            tgt_path = os.path.join(config.data_dir, f"{split}.{tgt_lang}")

            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                print(f"  [warn] Missing {src_path} or {tgt_path}, skip {src_lang}→{tgt_lang}")
                continue

            # Build byte-offset index (cheap — only stores ints)
            if src_lang not in self.offsets:
                self.offsets[src_lang] = _build_line_index(src_path)
                self.file_paths[src_lang] = src_path
            if tgt_lang not in self.offsets:
                self.offsets[tgt_lang] = _build_line_index(tgt_path)
                self.file_paths[tgt_lang] = tgt_path

            n_lines = min(len(self.offsets[src_lang]), len(self.offsets[tgt_lang]))
            cap = config.max_train_samples if split == "train" else n_lines
            for i in range(min(n_lines, cap)):
                self.samples.append((i, src_lang, tgt_lang, tag))

        full = len(self.samples)

        # Deterministic shuffle + slice to skip already-consumed data instantly
        if split == "train" and skip_samples > 0:
            rng = random.Random(42)
            rng.shuffle(self.samples)
            self.samples = self.samples[skip_samples:]

        print(f"  [{split}] {full} pairs → {len(self.samples)} after skip="
              f"{skip_samples} ({len(self.offsets)} files, "
              f"~{sum(len(o) for o in self.offsets.values()):,} lines indexed)")

    def _read_line(self, lang: str, line_no: int) -> str:
        offset = self.offsets[lang][line_no]
        with open(self.file_paths[lang], "r", encoding="utf-8") as f:
            f.seek(offset)
            return f.readline().strip()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        line_no, src_lang, tgt_lang, tag = self.samples[idx]
        src_text = f"{tag} {self._read_line(src_lang, line_no)}"
        tgt_text = self._read_line(tgt_lang, line_no)

        src_enc = self.tokenizer(
            src_text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )

        src_ids = src_enc["input_ids"].squeeze(0)
        src_mask = src_enc["attention_mask"].squeeze(0)
        tgt_ids = tgt_enc["input_ids"].squeeze(0)

        sos_id = self.tokenizer.cls_token_id or 101
        decoder_input = torch.cat([torch.tensor([sos_id]), tgt_ids[:-1]])
        tgt_pad_mask = (tgt_ids == self.tokenizer.pad_token_id)

        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_ids": tgt_ids,
            "decoder_input": decoder_input,
            "tgt_pad_mask": tgt_pad_mask,
        }


def collate_fn(batch):
    src_ids = torch.stack([item["src_ids"] for item in batch])
    src_mask = torch.stack([item["src_mask"] for item in batch])
    tgt_ids = torch.stack([item["tgt_ids"] for item in batch])
    decoder_input = torch.stack([item["decoder_input"] for item in batch])
    tgt_pad_mask = torch.stack([item["tgt_pad_mask"] for item in batch])
    return src_ids, src_mask, tgt_ids, decoder_input, tgt_pad_mask


def create_dataloaders(config, skip_samples: int = 0):
    # Auto-detect pre-tokenized data
    tokenized_dir = config.tokenized_dir or os.path.join(config.data_dir, "tokenized")
    use_pre = os.path.isdir(tokenized_dir) and any(
        f.endswith(".pt") for f in os.listdir(tokenized_dir))

    if use_pre:
        from train.pretokenized_dataset import PreTokenizedDataset
        train_dataset = PreTokenizedDataset(tokenized_dir, split="train",
                                            skip_samples=skip_samples)
        valid_dataset = PreTokenizedDataset(tokenized_dir, split="valid")
        train_shuffle = False  # dataset pre-shuffled deterministically
        train_shuffle = False  # dataset pre-shuffled deterministically
    else:
        train_dataset = TranslationDataset(config, split="train", skip_samples=skip_samples)
        valid_dataset = TranslationDataset(config, split="valid")
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=train_shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    return train_loader, valid_loader
