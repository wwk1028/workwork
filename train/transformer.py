"""Transformer encoder-decoder for machine translation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src_emb: torch.Tensor, src_mask: torch.Tensor):
        """
        Args:
            src_emb: (B, L, d_model)  BERT-embedded source tokens
            src_mask: (B, L)  padding mask (True = pad)
        Returns:
            memory: (B, L, d_model)
        """
        src_emb = self.pos_encoder(src_emb)
        # nn.Transformer expects src_key_padding_mask where True = ignore
        return self.encoder(src_emb, src_key_padding_mask=src_mask)


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = TransformerDecoderStack(decoder_layer, num_layers)

    def forward(self, tgt_emb: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor, memory_mask: torch.Tensor,
                tgt_pad_mask: torch.Tensor):
        """
        Args:
            tgt_emb: (B, L_tgt, d_model)
            memory: (B, L_src, d_model) from encoder
            tgt_mask: (L_tgt, L_tgt) causal mask
            memory_mask: (B, L_src)
            tgt_pad_mask: (B, L_tgt)
        Returns:
            output: (B, L_tgt, d_model)
        """
        tgt_emb = self.pos_encoder(tgt_emb)
        return self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )


class TransformerDecoderLayer(nn.Module):
    """Pre-LN decoder layer compatible with batch_first=True."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_key_padding_mask=None,
                tgt_key_padding_mask=None):
        # Self-attention
        x = self.norm1(tgt)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                      key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(attn_out)

        # Cross-attention
        x = self.norm2(tgt)
        attn_out, _ = self.cross_attn(x, memory, memory,
                                       key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(attn_out)

        # FFN
        x = self.norm3(tgt)
        tgt = tgt + self.dropout(self.ff(x))
        return tgt


class TransformerDecoderStack(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=decoder_layer.self_attn.embed_dim,
                nhead=decoder_layer.self_attn.num_heads,
                dim_feedforward=decoder_layer.ff[0].out_features,
                dropout=decoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask,
                        memory_key_padding_mask, tgt_key_padding_mask)
        return tgt
