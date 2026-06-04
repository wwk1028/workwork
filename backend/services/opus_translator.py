"""Translation service using OPUS-MT model (HuggingFace MarianMT).

Bidirectional (zh↔en) via language-tag fine-tuning.
Uses MarianMTModel with built-in beam search.
"""

import time
import os

import torch
import threading
from pathlib import Path

from transformers import MarianMTModel, MarianTokenizer


class OpusTranslatorService:
    """Per-instance MarianMTModel loader (one instance per model directory)."""

    _instances = {}  # model_dir → instance

    def __new__(cls, model_dir: str = None, device: str = None):
        if model_dir is None:
            model_dir = os.getenv("OPUS_MODEL_DIR", "opus_finetuned_full")
        model_dir = os.path.abspath(model_dir)
        if model_dir not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[model_dir] = inst
        return cls._instances[model_dir]

    def __init__(self, model_dir: str = None, device: str = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or os.getenv(
            "OPUS_MODEL_DIR", "opus_finetuned_full"
        )

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_models(self):
        if self._loaded:
            return

        path = Path(self.model_dir)
        if not path.exists():
            raise FileNotFoundError(f"OPUS model not found: {path}")

        print(f"[OpusTranslator] Loading from {path} ...")
        self.tokenizer = MarianTokenizer.from_pretrained(str(path))
        self.model = MarianMTModel.from_pretrained(str(path)).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"[OpusTranslator] Loaded on {self.device}, "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params")

    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def translate(self, text: str, direction: str = "zh",
                  num_beams: int = 6, max_length: int = 512) -> dict:
        """Translate text in the model's native direction only."""
        if not self._loaded:
            self._load_models()

        # Model is single-direction; determine label from tokenizer config
        dir_label = "zh2en" if self.tokenizer.target_lang == "eng" else "en2zh"

        t0 = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=max_length).to(self.device)

        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
        )
        translation = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        latency = (time.perf_counter() - t0) * 1000

        return {
            "source": text,
            "translation": translation.strip(),
            "direction": dir_label,
            "score": 0.0,
            "latency_ms": round(latency, 2),
        }

    def translate_batch(self, texts: list, direction: str = "zh",
                        num_beams: int = 6) -> list:
        """Batch translate in the model's native direction."""
        if not self._loaded:
            self._load_models()

        dir_label = "zh2en" if self.tokenizer.target_lang == "eng" else "en2zh"

        t0 = time.perf_counter()
        inputs = self.tokenizer(texts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=512).to(self.device)

        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=512,
            early_stopping=True,
        )
        translations = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        total_latency = (time.perf_counter() - t0) * 1000

        return [
            {
                "source": src,
                "translation": tgt.strip(),
                "direction": dir_label,
                "score": 0.0,
                "latency_ms": round(total_latency / len(texts), 2),
            }
            for src, tgt in zip(texts, translations)
        ]

    def get_info(self) -> dict:
        return {
            "model_name": "opus-mt-zh-en-finetuned",
            "directions": ["zh2en", "en2zh"],
            "beam_size": 6,
            "max_length": 512,
            "device": self.device,
        }
