"""Translation service — loads model, caches, handles inference."""

import time
import torch
import threading
from pathlib import Path

# Add project root to path for train imports + checkpoint pickle resolution
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train.config import Config
from train.seq2seq import Seq2Seq
from train.bert_embedding import BERTEmbedding
from train.beam_search import BeamSearch


class TranslatorService:
    """Singleton translation service with lazy model loading."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, zh2en_ckpt: str = None, en2zh_ckpt: str = None,
                 device: str = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.zh2en_ckpt = zh2en_ckpt
        self.en2zh_ckpt = en2zh_ckpt

        self.zh2en_model = None
        self.en2zh_model = None
        self.tokenizer = None
        self.beam = None
        self._loaded = False

    @classmethod
    def from_checkpoints(cls, zh2en: str, en2zh: str, device: str = None):
        inst = cls(zh2en_ckpt=zh2en, en2zh_ckpt=en2zh, device=device)
        inst._load_models()
        return inst

    def _load_models(self):
        if self._loaded:
            return
        tokenizer = BERTEmbedding("bert-base-multilingual-cased", 256).tokenizer

        self.zh2en_model = self._load_one(self.zh2en_ckpt)
        self.en2zh_model = self._load_one(self.en2zh_ckpt)
        self.tokenizer = tokenizer
        self.beam = BeamSearch(
            beam_size=5, max_len=128, length_penalty=1.0,
            sos_idx=tokenizer.cls_token_id or 101,
            eos_idx=tokenizer.sep_token_id or 102,
            pad_idx=tokenizer.pad_token_id or 0,
        )
        self._loaded = True
        print(f"[TranslatorService] Models loaded on {self.device}")

    def _load_one(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            return None
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", Config())
        cfg.device = self.device
        if not hasattr(cfg, "lang_tags") or cfg.lang_tags is None:
            cfg.lang_tags = {"zh": "<2zh>", "en": "<2en>"}
        model = Seq2Seq(cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def is_ready(self) -> bool:
        return self._loaded and self.zh2en_model is not None and self.en2zh_model is not None

    @torch.no_grad()
    def translate(self, text: str, direction: str, beam_size: int = 5) -> dict:
        """Translate a single sentence."""
        if not self._loaded:
            self._load_models()
        if direction == "zh":
            model = self.en2zh_model
            tag = "<2zh>"
        else:
            model = self.zh2en_model
            tag = "<2en>"
        if model is None:
            raise ValueError(f"No model loaded for direction {direction}")

        self.beam.beam_size = beam_size
        src_text = f"{tag} {text}"
        enc = self.tokenizer(src_text, max_length=128, truncation=True,
                             padding="max_length", return_tensors="pt")
        src_ids = enc["input_ids"].to(self.device)
        src_mask = enc["attention_mask"].to(self.device)

        t0 = time.perf_counter()
        hypotheses = self.beam.search(model, src_ids, src_mask)
        latency = (time.perf_counter() - t0) * 1000

        best_tokens = [t for t in hypotheses[0].tokens
                       if t not in (self.beam.sos_idx, self.beam.eos_idx, self.beam.pad_idx)]
        translation = self.tokenizer.decode(best_tokens, skip_special_tokens=True)
        return {
            "source": text,
            "translation": translation.strip(),
            "direction": direction,
            "score": hypotheses[0].score,
            "latency_ms": round(latency, 2),
        }

    @torch.no_grad()
    def translate_batch(self, texts: list, direction: str,
                        beam_size: int = 5) -> list:
        """Batch translate. Falls back to sequential for simplicity."""
        return [self.translate(t, direction, beam_size) for t in texts]

    def get_info(self) -> dict:
        return {
            "model_name": "seq2seq-transformer-bert",
            "directions": ["zh2en", "en2zh"],
            "beam_size": 5,
            "max_len": 128,
            "device": self.device,
        }
