"""Configuration for the Seq2Seq machine translation model.

Supports bidirectional (or multidirectional) translation via language tags.
Source sentences are prefixed with a target-language tag so the model learns
to translate in any configured direction, e.g.:

    "<2zh> Hello world"  →  "你好世界"
    "<2en> 你好世界"       →  "Hello world"
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class Config:
    # BERT
    bert_model_name: str = "bert-base-multilingual-cased"
    bert_local_path: str = ""
    bert_freeze: bool = True          # freeze BERT to save ~2GB VRAM
    bert_embed_dim: int = 768

    # Transformer (tuned for 8GB GPU)
    d_model: int = 256                # 512 → 256
    nhead: int = 8
    num_encoder_layers: int = 3       # 6 → 3
    num_decoder_layers: int = 3       # 6 → 3
    dim_feedforward: int = 1024       # 2048 → 1024
    dropout: float = 0.1

    # Beam Search
    beam_size: int = 5
    max_decode_len: int = 128
    length_penalty: float = 1.0

    # Vocabulary
    pad_idx: int = 0
    sos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    # Training
    batch_size: int = 4               # 32 → 4
    grad_accum_steps: int = 8         # effective batch = 4 * 8 = 32
    lr: float = 1e-4
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    warmup_steps: int = 4000
    max_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    use_amp: bool = True              # mixed-precision fp16

    # --- Bidirectional data ---
    # language tag map:  "zh" -> "<2zh>", "en" -> "<2en>"
    lang_tags: dict = field(default_factory=lambda: {"zh": "<2zh>", "en": "<2en>"})

    # Pairs to train, e.g. [("en", "zh"), ("zh", "en")]
    lang_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [("en", "zh"), ("zh", "en")])

    max_len: int = 256
    max_train_samples: int = 2000000  # max pairs per direction (2M → 4M samples)
    data_dir: str = "./data"
    tokenized_dir: str = ""  # pre-tokenized .pt shards (empty = use raw text)
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 1
    save_steps: int = 5000  # save checkpoint every N steps mid-epoch

    # Device — will auto-detect CUDA, fallback to cpu
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Pre-trained model path (for inference)
    model_path: Optional[str] = None

    # Resume from checkpoint (path to .pt file, or None to train from scratch)
    resume_from: Optional[str] = "checkpoints/mid_epoch.pt"
    # Override skip_steps when resuming (0 = use checkpoint value)
    resume_skip_steps: int = 0
