from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
from ..database import get_db
from ..models import User, Terminology
from ..auth import get_current_user

router = APIRouter(prefix="/terms", tags=["术语库"])


class TermCreate(BaseModel):
    source_term: str
    target_term: str


class TermResponse(BaseModel):
    id: int
    source_term: str
    target_term: str
    created_at: datetime
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[TermResponse])
async def get_terms(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    terms = db.query(Terminology).filter(
        Terminology.user_id == current_user.id
    ).order_by(Terminology.created_at.desc()).all()
    return terms


@router.post("/", response_model=TermResponse, status_code=201)
async def create_term(
    term_data: TermCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    existing = db.query(Terminology).filter(
        Terminology.user_id == current_user.id,
        Terminology.source_term == term_data.source_term
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="该术语已存在")
    
    new_term = Terminology(
        user_id=current_user.id,
        source_term=term_data.source_term,
        target_term=term_data.target_term
    )
    db.add(new_term)
    db.commit()
    db.refresh(new_term)
    
    return new_term


@router.delete("/{term_id}")
async def delete_term(
    term_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    term = db.query(Terminology).filter(
        Terminology.id == term_id,
        Terminology.user_id == current_user.id
    ).first()
    
    if not term:
        raise HTTPException(status_code=404, detail="术语不存在")
    
    db.delete(term)
    db.commit()
    
    return {"message": "删除成功"}
