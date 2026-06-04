"""SQLAlchemy ORM models — matches full schema."""

from datetime import datetime
from sqlalchemy import (Column, Integer, String, Float, DateTime, Text,
                        Boolean, Enum, ForeignKey, Index, Table)
from sqlalchemy.orm import relationship
from backend.database import Base

# ── 关联表 ────────────────────────────────────────────────────

user_role = Table(
    "user_role", Base.metadata,
    Column("user_id", Integer, ForeignKey("user.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("role.id"), primary_key=True),
)

# ── User ──────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    roles = relationship("Role", secondary=user_role, back_populates="users")

    def to_dict(self):
        return {
            "id": self.id, "username": self.username,
            "email": self.email, "full_name": self.full_name,
            "is_active": self.is_active,
            "roles": [r.name for r in self.roles],
            "created_at": fmt(self.created_at),
        }


# ── Role ──────────────────────────────────────────────────────

class Role(Base):
    __tablename__ = "role"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))

    users = relationship("User", secondary=user_role, back_populates="roles")


# ── Translation History ────────────────────────────────────────

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    source_text = Column(Text, nullable=False)
    target_text = Column(Text, nullable=False)
    source_lang = Column(String(10), nullable=False)
    target_lang = Column(String(10), nullable=False)
    model_used = Column(String(50))
    request_duration_ms = Column(Integer)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_user_id", "user_id"),
        Index("idx_created_at", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id, "user_id": self.user_id,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "model_used": self.model_used,
            "request_duration_ms": self.request_duration_ms,
            "ip_address": self.ip_address,
            "created_at": fmt(self.created_at),
        }


# ── Term ──────────────────────────────────────────────────────

class Term(Base):
    __tablename__ = "term"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    source_term = Column(String(255), nullable=False)
    target_term = Column(String(255), nullable=False)
    source_lang = Column(String(10), nullable=False)
    target_lang = Column(String(10), nullable=False)
    domain = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        Index("idx_term_user", "user_id"),
        Index("idx_term_langs", "source_lang", "target_lang"),
    )

    def to_dict(self):
        return {
            "id": self.id, "user_id": self.user_id,
            "source_term": self.source_term,
            "target_term": self.target_term,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "domain": self.domain,
            "created_at": fmt(self.created_at),
        }


# ── Task ──────────────────────────────────────────────────────

class Task(Base):
    __tablename__ = "task"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_uuid = Column(String(64), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    task_type = Column(Enum("document", "batch"), nullable=False)
    status = Column(Enum("pending", "processing", "completed", "failed"),
                    default="pending")
    input_file_path = Column(String(500))
    output_file_path = Column(String(500))
    source_lang = Column(String(10))
    target_lang = Column(String(10))
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "task_uuid": self.task_uuid,
            "user_id": self.user_id, "task_type": self.task_type,
            "status": self.status,
            "input_file_path": self.input_file_path,
            "output_file_path": self.output_file_path,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "error_message": self.error_message,
            "created_at": fmt(self.created_at),
            "updated_at": fmt(self.updated_at),
        }


# ── helper ────────────────────────────────────────────────────

def fmt(dt):
    return dt.isoformat() if dt else None
