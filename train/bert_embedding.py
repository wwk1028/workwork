"""BERT embedding layer for Seq2Seq machine translation.

Supports:
  - Auto-download from HuggingFace (default)
  - Loading from a local directory (set bert_local_path in config)
  - HuggingFace mirror via HF_ENDPOINT env var (e.g. https://hf-mirror.com)
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


def load_bert(model_name: str, local_path: str = ""):
    """Load BERT model + tokenizer. Uses local path when available."""
    if local_path and os.path.isdir(local_path):
        tokenizer = BertTokenizer.from_pretrained(local_path)
        model = BertModel.from_pretrained(local_path)
        return model, tokenizer
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        return model, tokenizer


class BERTEmbedding(nn.Module):
    """Wraps a pre-trained multilingual BERT to produce token embeddings.

    A linear projection maps from BERT's 768-dim space to the model's d_model.
    """

    def __init__(self, model_name: str, d_model: int,
                 freeze: bool = False, local_path: str = ""):
        super().__init__()
        self.bert, self.tokenizer = load_bert(model_name, local_path)
        self.proj = nn.Linear(self.bert.config.hidden_size, d_model)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self.bert.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id or 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(outputs.last_hidden_state)


class BERTVocabEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        return self.proj(x)
