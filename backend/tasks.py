"""Dramatiq actors for async translation tasks."""

import logging
import uuid
from io import BytesIO
from pathlib import Path

import dramatiq

# Import to register the broker
import backend.dramatiq_app  # noqa: F401
from backend.database import SessionLocal
from backend.models.db_models import Task
from backend.services.translator import TranslatorService

logger = logging.getLogger(__name__)


def _get_svc():
    """Get or create translator service (lazy singleton)."""
    return TranslatorService(
        zh2en_ckpt="checkpoints/zh2en/best_model.pt",
        en2zh_ckpt="checkpoints/en2zh/best_model.pt",
    )


def _update_task(task_uuid: str, **kwargs):
    """Update task fields in DB."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.task_uuid == task_uuid).first()
        if task:
            for k, v in kwargs.items():
                setattr(task, k, v)
            db.commit()
    finally:
        db.close()


def _read_file(path: str) -> str:
    """Read text content from a file, supporting txt/docx/pdf."""
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return Path(path).read_text(encoding="utf-8")
    elif ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _write_file(path: str, content: str):
    """Write translated content, preserving format for docx."""
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        from docx import Document
        doc = Document()
        for line in content.split("\n"):
            if line.strip():
                doc.add_paragraph(line)
        doc.save(path)
    else:
        Path(path).write_text(content, encoding="utf-8")


@dramatiq.actor(max_retries=0)
def translate_document(task_uuid: str):
    """Translate a document file and write the translated output."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.task_uuid == task_uuid).first()
        if not task:
            logger.error("Task %s not found", task_uuid)
            return

        task.status = "processing"
        db.commit()

        svc = _get_svc()
        svc._load_models()

        text = _read_file(task.input_file_path)
        direction = "zh" if task.target_lang == "zh" else "en"
        results = svc.translate_batch(
            [s for s in text.split("\n") if s.strip()], direction=direction
        )
        output_text = "\n".join(r["translation"] for r in results)

        output_path = task.input_file_path + ".translated" + Path(task.input_file_path).suffix
        _write_file(output_path, output_text)

        task.status = "completed"
        task.output_file_path = output_path
        db.commit()
    except Exception as e:
        logger.exception("Document translation failed: %s", e)
        try:
            task = db.query(Task).filter(Task.task_uuid == task_uuid).first()
            if task:
                task.status = "failed"
                task.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
