from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from ..database import get_db
from ..models import User, TranslationHistory
from ..auth import get_current_user

router = APIRouter(prefix="/history", tags=["历史记录"])


class HistoryResponse(BaseModel):
    id: int
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    processing_time: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[HistoryResponse])
async def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    history = db.query(TranslationHistory).filter(
        TranslationHistory.user_id == current_user.id
    ).order_by(TranslationHistory.created_at.desc()).offset(skip).limit(limit).all()
    
    return history


@router.get("/{history_id}", response_model=HistoryResponse)
async def get_history_item(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    item = db.query(TranslationHistory).filter(
        TranslationHistory.id == history_id,
        TranslationHistory.user_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")
    
    return item


@router.delete("/{history_id}")
async def delete_history(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    item = db.query(TranslationHistory).filter(
        TranslationHistory.id == history_id,
        TranslationHistory.user_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="记录不存在")
    
    db.delete(item)
    db.commit()
    
    return {"message": "删除成功"}


@router.delete("/")
async def clear_all_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.query(TranslationHistory).filter(
        TranslationHistory.user_id == current_user.id
    ).delete()
    db.commit()
    
    return {"message": "清空成功"}
