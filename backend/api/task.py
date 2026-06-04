"""Task tracking endpoints for async translations."""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import Task, User
from backend.auth import get_current_user

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", status_code=201)
async def create_task(
    file: UploadFile = File(...),
    task_type: str = Form(..., regex="^(document|batch)$"),
    source_lang: str = Form(default="en"),
    target_lang: str = Form(default="zh"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    task_uuid = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".txt"
    save_path = UPLOAD_DIR / f"{task_uuid}{ext}"
    content = await file.read()
    save_path.write_bytes(content)

    task = Task(
        task_uuid=task_uuid,
        user_id=user.id,
        task_type=task_type,
        status="pending",
        input_file_path=str(save_path),
        source_lang=source_lang,
        target_lang=target_lang,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    from backend.tasks import translate_document
    translate_document.send(task_uuid)
    return task.to_dict()


@router.get("")
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str = Query(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Task).filter(Task.user_id == user.id)
    if status:
        q = q.filter(Task.status == status)
    total = q.count()
    items = q.order_by(Task.created_at.desc()) \
              .offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size,
            "items": [t.to_dict() for t in items]}


@router.get("/{task_uuid}")
async def get_task(task_uuid: str, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    task = db.query(Task).filter(
        Task.task_uuid == task_uuid, Task.user_id == user.id
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task.to_dict()
