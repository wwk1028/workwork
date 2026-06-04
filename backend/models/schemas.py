"""Pydantic schemas for request/response validation."""

from typing import List
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512,
                      description="Source text to translate")
    direction: str = Field(default="zh", pattern="^(zh|en)$",
                           description="Target language: 'zh' or 'en'")
    beam_size: int = Field(default=5, ge=1, le=20,
                           description="Beam search width")


class BatchTranslateRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100,
                             description="List of source texts")
    direction: str = Field(default="zh", pattern="^(zh|en)$")
    beam_size: int = Field(default=5, ge=1, le=20)


class TranslateResponse(BaseModel):
    source: str
    translation: str
    direction: str
    score: float
    latency_ms: float


class BatchTranslateResponse(BaseModel):
    results: List[TranslateResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    model_name: str
    directions: List[str]
    beam_size: int
    max_len: int
    device: str
