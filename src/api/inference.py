"""Model loading and prediction logic for the hallucination detection API.

The module tries to load the fine-tuned DistilBERT checkpoint at
``models/saved_distilbert_hallucination_model`` first. If transformers /
torch are not installed or the model directory is missing, it falls back to
a transparent keyword-based heuristic so the UI is usable while the heavy
ML dependencies are being installed.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label config - must match the mapping used in the notebook
# (none=0, fabrication=1, confusion=2)
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[int, str] = {0: "none", 1: "fabrication", 2: "confusion"}
ID2LABEL: Dict[int, str] = LABEL_MAP
LABEL2ID: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}
LABELS: List[str] = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

# Default checkpoint location (relative to the project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "saved_distilbert_hallucination_model"
MAX_LENGTH = 256


@dataclass
class Prediction:
    label: str
    label_id: int
    confidence: float
    scores: Dict[str, float]
    backend: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Optional heavy imports - wrapped so the API can boot without them.
# ---------------------------------------------------------------------------
def _try_import_transformers() -> Optional[Tuple]:
    try:
        import torch  # noqa: F401  (imported for side-effect typing)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # ImportError or any other boot failure
        logger.warning("transformers/torch unavailable: %s", exc)
        return None
    return AutoModelForSequenceClassification, AutoTokenizer, torch


# ---------------------------------------------------------------------------
# Heuristic backend - used when transformers is not available. The intent is
# to keep the UI demonstrable, not to reproduce model accuracy.
# ---------------------------------------------------------------------------
_FABRICATION_TERMS = {
    "invented", "made up", "made-up", "fictional", "imaginary", "pretend",
    "no such", "does not exist", "doesn't exist", "mythical", "fake",
    "hoax", "untrue", "fabricated",
}
_CONFUSION_TERMS = {
    "confused", "confusing", "mix-up", "mix up", "mistaken", "misinterpret",
    "misunderstanding", "ambiguous", "unclear", "tangled", "jumbled",
    "conflate", "confuses", "wrongly", "incorrectly", "mistakenly",
}
_QUESTION_HINTS = ("?", "what", "why", "how", "when", "where", "who", "which")


def _heuristic_predict(text: str) -> Prediction:
    start = time.perf_counter()
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    fab_hits = sum(1 for term in _FABRICATION_TERMS if term in lowered)
    con_hits = sum(1 for term in _CONFUSION_TERMS if term in lowered)
    has_question = any(h in lowered for h in _QUESTION_HINTS)

    # Score in [0, 1] with a small smoothing so the softmax is well-defined.
    none_score = 0.55 + (0.05 if has_question else 0.0)
    fab_score = 0.20 + 0.15 * min(fab_hits, 3)
    con_score = 0.20 + 0.15 * min(con_hits, 3)

    raw = {
        "none": none_score,
        "fabrication": fab_score,
        "confusion": con_score,
    }
    total = sum(raw.values()) or 1.0
    scores = {k: v / total for k, v in raw.items()}

    label_id = max(scores, key=scores.get)
    label_id_num = LABEL2ID[label_id]
    confidence = scores[label_id]

    latency = (time.perf_counter() - start) * 1000.0
    return Prediction(
        label=label_id,
        label_id=label_id_num,
        confidence=float(confidence),
        scores=scores,
        backend="heuristic",
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Transformer backend - loads the saved DistilBERT model if available.
# ---------------------------------------------------------------------------
class TransformerModel:
    def __init__(self, model_dir: Path) -> None:
        imports = _try_import_transformers()
        if imports is None:
            raise RuntimeError(
                "transformers/torch are not installed; cannot load transformer model."
            )
        AutoModelForSequenceClassification, AutoTokenizer, torch = imports
        self.torch = torch
        logger.info("Loading model from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        torch = self.torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
        label_id_num = int(max(range(len(probs)), key=lambda i: probs[i]))
        label = ID2LABEL[label_id_num]
        confidence = float(probs[label_id_num])

        latency = (time.perf_counter() - start) * 1000.0
        return Prediction(
            label=label,
            label_id=label_id_num,
            confidence=confidence,
            scores=scores,
            backend="transformer",
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Singleton accessor - loads the model once per process.
# ---------------------------------------------------------------------------
class ModelService:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model: Optional[TransformerModel] = None
        self._model_dir: Path = Path(
            model_dir or os.environ.get("HALLUCINATION_MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        self._tried = False

    def _ensure_loaded(self) -> Optional[TransformerModel]:
        if self._tried:
            return self._model
        self._tried = True
        if not self._model_dir.exists():
            logger.info(
                "Model directory %s not found; using heuristic backend.",
                self._model_dir,
            )
            return None
        try:
            self._model = TransformerModel(self._model_dir)
        except Exception as exc:  # broad: any load failure falls back safely
            logger.warning(
                "Failed to load transformer model from %s: %s. "
                "Falling back to heuristic backend.",
                self._model_dir,
                exc,
            )
            self._model = None
        return self._model

    @property
    def backend(self) -> str:
        return "transformer" if self._ensure_loaded() else "heuristic"

    @property
    def is_loaded(self) -> bool:
        return self._ensure_loaded() is not None

    def predict(self, text: str) -> Prediction:
        model = self._ensure_loaded()
        if model is not None:
            return model.predict(text)
        return _heuristic_predict(text)


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService()
