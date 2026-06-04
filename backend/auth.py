"""JWT authentication utilities."""

import os
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import User

SECRET_KEY = os.getenv("JWT_SECRET", "mt-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid token")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="user not found or inactive")
    return user


def require_role(*roles: str):
    """Dependency factory: restrict endpoint to specific roles."""
    def checker(user: User = Depends(get_current_user)):
        user_roles = {r.name for r in user.roles}
        if not user_roles & set(roles):
            raise HTTPException(status_code=403, detail="insufficient permissions")
        return user
    return checker
