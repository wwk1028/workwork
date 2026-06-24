"""Seq2Seq model combining BERT embeddings + Transformer + output projection."""

import torch
import torch.nn as nn

from train.bert_embedding import BERTEmbedding
from train.transformer import TransformerEncoder, TransformerDecoder


class Seq2Seq(nn.Module):
    """End-to-end Seq2Seq model for machine translation.

    Encoder: BERT embedding (frozen or fine-tuned) → TransformerEncoder
    Decoder: learned target embedding → TransformerDecoder → output projection
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- BERT source embedding ---
        self.src_embed = BERTEmbedding(
            model_name=config.bert_model_name,
            d_model=config.d_model,
            freeze=config.bert_freeze,
            local_path=config.bert_local_path
        )

        # Derive target vocab size from the BERT tokenizer so indices always match
        self.tgt_vocab_size = self.src_embed.tokenizer.vocab_size

        # --- Target token embedding (learned from scratch) ---
        self.tgt_embed = nn.Embedding(
            self.tgt_vocab_size, config.d_model,
            padding_idx=config.pad_idx
        )

        # --- Transformer ---
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )

        self.decoder = TransformerDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )

        # --- Output projection to BERT vocabulary ---
        self.output_head = nn.Linear(config.d_model, self.tgt_vocab_size)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor):
        """Encoder forward pass.

        Returns:
            memory: (B, L_src, d_model)
            memory_pad_mask: (B, L_src)  True = pad position
        """
        src_emb = self.src_embed(src_ids, src_mask)
        pad_mask = ~src_mask.bool()
        memory = self.encoder(src_emb, pad_mask)
        return memory, pad_mask

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
               memory_pad_mask: torch.Tensor, tgt_pad_mask: torch.Tensor = None):
        """Decoder forward pass (teacher-forcing).

        Args:
            tgt_ids: (B, L_tgt)  shifted right, BERT token IDs
            memory: (B, L_src, d_model)
            memory_pad_mask: (B, L_src)  True = pad
            tgt_pad_mask: (B, L_tgt)  True = pad
        Returns:
            logits: (B, L_tgt, tgt_vocab_size)
        """
        B, L_tgt = tgt_ids.shape
        device = tgt_ids.device

        tgt_emb = self.tgt_embed(tgt_ids)

        causal_mask = torch.triu(
            torch.full((L_tgt, L_tgt), float("-inf"), device=device), diagonal=1
        )

        output = self.decoder(tgt_emb, memory, causal_mask,
                              memory_pad_mask, tgt_pad_mask)
        return self.output_head(output)

    def decode_step(self, memory: torch.Tensor, memory_pad_mask: torch.Tensor,
                    tgt_ids: torch.Tensor):
        """Single-step decode for beam search. Returns logits for last position."""
        logits = self.decode(tgt_ids, memory, memory_pad_mask)
        return logits[:, -1:, :]

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor,
                tgt_ids: torch.Tensor, tgt_pad_mask: torch.Tensor = None):
        """Full forward pass.

        Args:
            src_ids: (B, L_src)  BERT input ids
            src_mask: (B, L_src)  1 = token, 0 = pad
            tgt_ids: (B, L_tgt)  decoder input (shifted right)
        Returns:
            logits: (B, L_tgt, tgt_vocab_size)
        """
        memory, memory_pad_mask = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, memory, memory_pad_mask, tgt_pad_mask)


class Seq2SeqLoss(nn.Module):
    def __init__(self, pad_idx: int, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=label_smoothing
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, L, V = logits.shape
        logits = logits.reshape(B * L, V)
        targets = targets.reshape(B * L)
        return self.criterion(logits, targets)
