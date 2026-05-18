from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.routers import auth, translate, terms, history

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="基于Transformer的中英翻译系统API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(translate.router, prefix=settings.API_PREFIX)
app.include_router(terms.router, prefix=settings.API_PREFIX)
app.include_router(history.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "message": "欢迎使用中英翻译系统API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
