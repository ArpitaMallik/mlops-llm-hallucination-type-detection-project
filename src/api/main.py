"""FastAPI application exposing the hallucination-type classifier.

Run locally with:

    uvicorn src.api.main:app --reload --port 8000

or via the convenience script at the repo root:

    python run_api.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.inference import LABELS, get_service
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hallucination_api")

# Paths ----------------------------------------------------------------------
API_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# App -----------------------------------------------------------------------
app = FastAPI(
    title="LLM Hallucination Type Detector",
    description=(
        "Multi-class classifier that predicts whether a prompt's expected "
        "LLM response is `none`, `fabrication`, or `confusion`."
    ),
    version="0.1.0",
)


@app.on_event("startup")
def _warm_up_model() -> None:
    """Trigger lazy model load at startup so the first request is fast."""
    try:
        backend = get_service().backend
        logger.info("Inference backend on startup: %s", backend)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Model warm-up failed: %s", exc)


# Static files (optional, served if the directory exists) -------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Routes --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index(request: Request) -> HTMLResponse:
    """Render the single-page UI."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "labels": LABELS,
            "backend": get_service().backend,
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="ok",
        model_backend=service.backend,
        model_loaded=service.is_loaded,
        labels=LABELS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_json(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")

    pred = get_service().predict(text)
    return PredictResponse(
        label=pred.label,
        label_id=pred.label_id,
        confidence=pred.confidence,
        scores=pred.scores if payload.return_scores else None,
        model_backend=pred.backend,
        latency_ms=pred.latency_ms,
    )


@app.post("/predict/form", response_class=HTMLResponse, tags=["inference"])
def predict_form(
    request: Request,
    text: str = Form(...),
) -> HTMLResponse:
    """Form endpoint used by the HTML UI to render the result inline."""
    text = (text or "").strip()
    if not text:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "labels": LABELS,
                "backend": get_service().backend,
                "error": "Please enter a prompt to classify.",
                "input_text": text,
            },
            status_code=400,
        )

    pred = get_service().predict(text)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "labels": LABELS,
            "backend": pred.backend,
            "result": {
                "label": pred.label,
                "label_id": pred.label_id,
                "confidence": pred.confidence,
                "scores": pred.scores,
                "latency_ms": pred.latency_ms,
            },
            "input_text": text,
        },
    )


# Convenience error handler for unhandled exceptions -----------------------
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error": str(exc)},
    )
