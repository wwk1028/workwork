"""Terminology management endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import Term, User
from backend.auth import get_current_user

router = APIRouter(prefix="/api/v1/terms", tags=["terms"])


class TermCreate(BaseModel):
    source_term: str = Field(..., min_length=1, max_length=255)
    target_term: str = Field(..., min_length=1, max_length=255)
    source_lang: str = Field(default="en", max_length=10)
    target_lang: str = Field(default="zh", max_length=10)
    domain: Optional[str] = None


class TermUpdate(BaseModel):
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    domain: Optional[str] = None


@router.get("")
async def list_terms(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    domain: str = Query(""),
    source_lang: str = Query(""),
    target_lang: str = Query(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Term)
    if user is None or not any(r.name == "admin" for r in user.roles):
        q = q.filter((Term.user_id == None) | (Term.user_id == user.id))
    if domain:
        q = q.filter(Term.domain == domain)
    if source_lang:
        q = q.filter(Term.source_lang == source_lang)
    if target_lang:
        q = q.filter(Term.target_lang == target_lang)

    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size,
            "items": [t.to_dict() for t in items]}


@router.post("")
async def create_term(req: TermCreate, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    term = Term(
        user_id=user.id, source_term=req.source_term,
        target_term=req.target_term, source_lang=req.source_lang,
        target_lang=req.target_lang, domain=req.domain,
    )
    db.add(term)
    db.commit()
    db.refresh(term)
    return term.to_dict()


@router.put("/{term_id}")
async def update_term(term_id: int, req: TermUpdate,
                      db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    term = db.query(Term).filter(Term.id == term_id, Term.user_id == user.id).first()
    if not term:
        raise HTTPException(status_code=404, detail="term not found")
    if req.source_term is not None:
        term.source_term = req.source_term
    if req.target_term is not None:
        term.target_term = req.target_term
    if req.domain is not None:
        term.domain = req.domain
    db.commit()
    return term.to_dict()


@router.delete("/{term_id}")
async def delete_term(term_id: int, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    term = db.query(Term).filter(Term.id == term_id, Term.user_id == user.id).first()
    if not term:
        raise HTTPException(status_code=404, detail="term not found")
    db.delete(term)
    db.commit()
    return {"deleted": term_id}
