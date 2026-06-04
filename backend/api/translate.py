"""Translation API endpoints — REST + WebSocket + DB persistence."""

import os
import time
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.schemas import (
    TranslateRequest, TranslateResponse,
    BatchTranslateRequest, BatchTranslateResponse, ModelInfo,
)
from backend.models.db_models import TranslationHistory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["translate"])

# ── Hybrid: OPUS (zh→en) + scratch (en→zh) ──
OPUS_DIR = os.getenv("OPUS_MODEL_DIR", "opus_finetuned_full")
_OPUS_READY = os.path.isdir(OPUS_DIR) and os.path.isfile(
    os.path.join(OPUS_DIR, "model.safetensors")
)

_zh2en_svc = None
_en2zh_svc = None


def _init_services():
    global _zh2en_svc, _en2zh_svc
    if _zh2en_svc is None:
        from backend.services.opus_translator import OpusTranslatorService
        _zh2en_svc = OpusTranslatorService(model_dir=OPUS_DIR)
        logger.info(f"zh→en: OPUS-MT ({OPUS_DIR})")

    if _en2zh_svc is None:
        en2zh_dir = os.getenv("EN2ZH_MODEL_DIR", "opus_en2zh")
        from backend.services.opus_translator import OpusTranslatorService
        _en2zh_svc = OpusTranslatorService(model_dir=en2zh_dir)
        logger.info(f"en→zh: OPUS-MT ({en2zh_dir})")


def _get_service(direction: str):
    _init_services()
    # direction = target language: "zh" → en→zh service, "en" → zh→en service
    if direction in ("zh", "en2zh", "en-zh"):
        return _en2zh_svc
    return _zh2en_svc


def _save_record(db: Session, result: dict, client_ip: str = "",
                 user_id: int = None):
    record = TranslationHistory(
        user_id=user_id,
        source_text=result["source"],
        target_text=result["translation"],
        source_lang="zh" if result.get("direction", "zh2en") in ("zh2en", "zh") else "en",
        target_lang="en" if result.get("direction", "zh2en") in ("zh2en", "zh") else "zh",
        model_used=result.get("model", "opus-mt" if _OPUS_READY else "seq2seq"),
        request_duration_ms=int(result.get("latency_ms", 0)),
        ip_address=client_ip,
    )
    db.add(record)
    db.commit()


# ── REST: health & info ───────────────────────────────────────


@router.get("/health")
async def health():
    _init_services()
    return {"status": "ok", "zh2en": _zh2en_svc is not None and _zh2en_svc.is_ready(),
            "en2zh": _en2zh_svc is not None and _en2zh_svc.is_ready()}


@router.get("/info", response_model=ModelInfo)
async def model_info():
    _init_services()
    return _zh2en_svc.get_info()


# ── REST: translate ────────────────────────────────────────────


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, db: Session = Depends(get_db)):
    svc = _get_service(req.direction)
    if not svc.is_ready():
        svc._load_models()
    result = svc.translate(req.text, req.direction, req.beam_size)
    _save_record(db, result)
    return TranslateResponse(**result)


@router.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(req: BatchTranslateRequest,
                           db: Session = Depends(get_db)):
    svc = _get_service(req.direction)
    if not svc.is_ready():
        svc._load_models()
    t0 = time.perf_counter()
    parsed = []
    for r in svc.translate_batch(req.texts, req.direction, req.beam_size):
        _save_record(db, r)
        parsed.append(TranslateResponse(**r))
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchTranslateResponse(results=parsed, total_latency_ms=round(total_ms, 2))


# ── REST: history ──────────────────────────────────────────────


@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    direction: str = Query("", description="筛选方向: zh2en / en2zh"),
    db: Session = Depends(get_db),
):
    q = db.query(TranslationHistory)
    if direction and direction in ("zh2en", "en2zh"):
        src = "zh" if direction == "zh2en" else "en"
        tgt = "en" if direction == "zh2en" else "zh"
        q = q.filter(TranslationHistory.source_lang == src,
                     TranslationHistory.target_lang == tgt)
    total = q.count()
    records = (q.order_by(TranslationHistory.created_at.desc())
               .offset((page - 1) * page_size)
               .limit(page_size)
               .all())
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [r.to_dict() for r in records],
    }


@router.delete("/history/{record_id}")
async def delete_history(record_id: int, db: Session = Depends(get_db)):
    record = db.query(TranslationHistory).filter(
        TranslationHistory.id == record_id
    ).first()
    if not record:
        return {"error": "record not found"}
    db.delete(record)
    db.commit()
    return {"deleted": record_id}


# ── WebSocket ──────────────────────────────────────────────────


@router.websocket("/translate/ws")
async def translate_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            direction = data.get("direction", "zh")
            beam_size = int(data.get("beam_size", 5))
            if not text:
                await websocket.send_json({"error": "empty text"})
                continue
            svc = _get_service(direction)
            if not svc.is_ready():
                svc._load_models()
            result = svc.translate(text, direction, beam_size)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
