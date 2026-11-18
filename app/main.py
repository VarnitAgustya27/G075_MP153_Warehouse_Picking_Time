"""
app/main.py
-----------
Production serving layer for MP153 – Warehouse Picking Time (G075).

Endpoints
---------
GET  /health   → {"status": "ok"}
POST /predict  → {"prediction": <float>}

Validation (via Pydantic Field, returns HTTP 422 on violation):
  • aisle_distance_m  ≤ 700   (client rule 1 — PROBLEM.md §5)
  • shelf_height_m    ≥ 0     (client rule 2 — PROBLEM.md §5)

Logging
-------
One structured JSON line per prediction written to stdout (no file, no PII).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured JSON logger – stdout only, no personal data
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Render every LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra structured fields the caller passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "taskName",
            }:
                payload[key] = value
        return json.dumps(payload, default=str)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


log = _build_logger("mp153.api")

# ---------------------------------------------------------------------------
# Model loader — HTTP 503 if model is unavailable
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"


def _load_model() -> Any | None:
    """Return the joblib model, or None when the file is missing / corrupt."""
    if not _MODEL_PATH.exists():
        log.warning(
            "Model file not found – /predict will return HTTP 503",
            extra={"model_path": str(_MODEL_PATH)},
        )
        return None
    try:
        model = joblib.load(_MODEL_PATH)
        log.info("Model loaded", extra={"model_path": str(_MODEL_PATH)})
        return model
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failed to load model – /predict will return HTTP 503",
            extra={"model_path": str(_MODEL_PATH), "error": str(exc)},
        )
        return None


_MODEL = _load_model()

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Exactly the 8 inference-time features from PROBLEM.md §5.
    Only the two constraints specified by the client are enforced here;
    all other fields accept any value of the correct type.
    """

    items_in_order: int = Field(
        ...,
        description="Number of items in the order.",
    )
    aisle_distance_m: float = Field(
        ...,
        le=700.0,
        description="Walking distance in metres. Must not exceed 700 m.",
    )
    picker_exp_months: int = Field(
        ...,
        description="Picker experience in months.",
    )
    shelf_height_m: float = Field(
        ...,
        ge=0.0,
        description="Average shelf height in metres. Must be ≥ 0.",
    )
    shift_hour: int = Field(
        ...,
        description="Hour of picking.",
    )
    order_weight_kg: float = Field(
        ...,
        description="Order weight in kg.",
    )
    zone_type: str = Field(
        ...,
        description="Storage zone (e.g. ambient, chilled, bulky).",
    )
    shift: str = Field(
        ...,
        description="Shift (e.g. Morning, Evening, Night).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "items_in_order": 5,
                "aisle_distance_m": 120.5,
                "picker_exp_months": 18,
                "shelf_height_m": 1.4,
                "shift_hour": 9,
                "order_weight_kg": 7.3,
                "zone_type": "ambient",
                "shift": "Morning",
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: float


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MP153 Warehouse Picking Time API",
    version="1.0",
    description="Predicts pick_seconds for a warehouse order (G075 / MP153).",
)

# Feature column order that the trained pipeline expects
_FEATURE_COLS = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "zone_type",
    "shift",
]


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Return predicted pick_seconds for one order.

    Returns HTTP 503 if the model is not loaded.
    Re-reads the module-level _MODEL on every call so unit tests can patch it.
    """
    import sys
    import pandas as pd

    _current_model = sys.modules[__name__]._MODEL
    if _current_model is None:
        log.error("Model unavailable – returning HTTP 503")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not available. The model file is missing or failed to load."},
        )

    t0 = time.perf_counter()

    row = {
        "items_in_order": body.items_in_order,
        "aisle_distance_m": body.aisle_distance_m,
        "picker_exp_months": body.picker_exp_months,
        "shelf_height_m": body.shelf_height_m,
        "shift_hour": body.shift_hour,
        "order_weight_kg": body.order_weight_kg,
        "zone_type": body.zone_type,
        "shift": body.shift,
    }

    try:
        X = pd.DataFrame([row])[_FEATURE_COLS]
        raw = _current_model.predict(X)
        prediction = float(raw[0])
    except Exception as exc:  # noqa: BLE001
        log.error("Prediction error", extra={"error": str(exc)})
        raise

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # --- Structured log line (no PII) ------------------------------------
    log.info(
        "prediction",
        extra={
            "items_in_order": body.items_in_order,
            "aisle_distance_m": body.aisle_distance_m,
            "picker_exp_months": body.picker_exp_months,
            "shelf_height_m": body.shelf_height_m,
            "shift_hour": body.shift_hour,
            "order_weight_kg": body.order_weight_kg,
            "zone_type": body.zone_type,
            "shift": body.shift,
            "prediction": prediction,
            "source": "model",
            "latency_ms": elapsed_ms,
        },
    )

    return PredictResponse(prediction=prediction)
