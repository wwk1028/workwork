"""Authentication endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import User, Role
from backend.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    email: Optional[str] = None
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="username exists")
    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        email=req.email,
        full_name=req.full_name,
    )
    # Default role: user
    default_role = db.query(Role).filter(Role.name == "user").first()
    if default_role:
        user.roles.append(default_role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "username": user.username, "message": "registered"}


@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="wrong username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="account disabled")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user": user.to_dict()}
