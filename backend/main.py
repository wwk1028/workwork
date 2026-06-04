"""FastAPI backend for the Seq2Seq translation service."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db, SessionLocal
from backend.models.db_models import Role
from backend.api.translate import router as translate_router
from backend.api.auth import router as auth_router
from backend.api.term import router as term_router
from backend.api.task import router as task_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _seed_roles():
    """Insert default roles if they don't exist."""
    db = SessionLocal()
    try:
        for name, desc in [("admin", "管理员"), ("term_manager", "术语管理员"),
                           ("user", "普通用户")]:
            if not db.query(Role).filter(Role.name == name).first():
                db.add(Role(name=name, description=desc))
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB + seed roles + preload models
    logger.info("Initializing database ...")
    init_db()
    _seed_roles()
    yield
    # Shutdown
    logger.info("Shutting down.")


app = FastAPI(
    title="Seq2Seq Translation API",
    version="1.0.0",
    description="Chinese ↔ English machine translation with BERT + Transformer + Beam Search",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translate_router)
app.include_router(auth_router)
app.include_router(term_router)
app.include_router(task_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
