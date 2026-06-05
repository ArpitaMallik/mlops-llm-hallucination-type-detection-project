"""Pydantic schemas for the hallucination detection API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """JSON request body for /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The prompt or LLM response text to classify.",
    )
    return_scores: bool = Field(
        default=True,
        description="Whether to include per-class probability scores in the response.",
    )


class PredictResponse(BaseModel):
    """Response payload for /predict."""

    label: str = Field(..., description="Predicted hallucination type.")
    label_id: int = Field(..., description="Numeric id of the predicted class.")
    confidence: float = Field(..., description="Confidence of the top prediction (0-1).")
    scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-class probability scores keyed by label name.",
    )
    model_backend: str = Field(
        ...,
        description="Which backend produced the prediction (transformer|heuristic).",
    )
    latency_ms: float = Field(..., description="Server-side inference time in ms.")


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    model_loaded: bool
    labels: List[str]
