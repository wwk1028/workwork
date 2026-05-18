from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import time
from ..database import get_db
from ..models import User, TranslationHistory
from ..auth import get_current_user

router = APIRouter(prefix="/translate", tags=["翻译"])


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    use_terminology: Optional[bool] = True


class TranslateResponse(BaseModel):
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    processing_time: float


def simple_translate(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang == "zh" and target_lang == "en":
        return f"[EN] {text}"
    elif source_lang == "en" and target_lang == "zh":
        return f"[中文] {text}"
    return text


@router.post("/", response_model=TranslateResponse)
async def translate(
    request: TranslateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    start_time = time.time()
    
    target_text = simple_translate(request.text, request.source_lang, request.target_lang)
    
    processing_time = time.time() - start_time
    
    history = TranslationHistory(
        user_id=current_user.id,
        source_text=request.text,
        target_text=target_text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        processing_time=processing_time
    )
    db.add(history)
    db.commit()
    
    return TranslateResponse(
        source_text=request.text,
        target_text=target_text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        processing_time=processing_time
    )
