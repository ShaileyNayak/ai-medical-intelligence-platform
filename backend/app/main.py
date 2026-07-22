"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import health_router, history_router, predict_router
from app.core.config import settings
from app.core.logging import new_request_id, setup_logging
from app.db.database import init_db
from app.services.inference_service import get_inference_service

setup_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.heatmap_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    get_inference_service()
    yield


app = FastAPI(
    title="AI Medical Intelligence Platform",
    version="1.0.0",
    description=(
        "Multi-disease imaging platform (chest X-ray, brain MRI, skin lesion) with "
        "per-module Grad-CAM and LLM-assisted reporting. Educational prototype — not a medical device."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    rid = new_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)

static_root = Path("static")
static_root.mkdir(exist_ok=True)
(Path(settings.upload_dir)).mkdir(parents=True, exist_ok=True)
(Path(settings.heatmap_dir)).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
