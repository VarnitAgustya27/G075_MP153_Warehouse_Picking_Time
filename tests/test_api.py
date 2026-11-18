"""
tests/test_api.py
-----------------
Pytest suite for app/main.py (G075 · MP153 – Warehouse Picking Time).

Uses FastAPI's TestClient (httpx + ASGI) for all HTTP-level tests, plus
direct Pydantic schema checks for the two client validation rules.

Covers
------
GET /health
  • Returns HTTP 200
  • Body is {"status": "ok"}

POST /predict – happy path
  • Returns HTTP 200 with a numeric {"prediction": <float>} when model present
  • aisle_distance_m = 700.0 (boundary) → 200
  • shelf_height_m = 0.0 (boundary) → 200

POST /predict – client validation rules (HTTP 422)
  • aisle_distance_m = 750  → 422
  • shelf_height_m   = -1.5 → 422
  • aisle_distance_m = 700.01 → 422
  • Missing required field → 422
  • Empty body → 422
  • Wrong type → 422

POST /predict – model unavailable (HTTP 503)
  • Returns HTTP 503 with a "detail" key when _MODEL is None
  • Validation (422) still fires before the 503 check

Pydantic schema (field-level, no event loop needed)
  • All 8 fields present and typed correctly
  • aisle_distance_m > 700 → ValidationError with less_than_equal
  • shelf_height_m < 0     → ValidationError with greater_than_equal
  • Each of the 8 fields is individually required

Structured JSON logging
  • One JSON line emitted per successful prediction
  • Line contains "prediction" key with a numeric value
  • No "record_id" PII field
  • Logger has no FileHandler
  • Logger StreamHandler targets stdout
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as _main
from app.main import PredictRequest, PredictResponse, app

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict = {
    "items_in_order": 5,
    "aisle_distance_m": 120.5,
    "picker_exp_months": 18,
    "shelf_height_m": 1.4,
    "shift_hour": 9,
    "order_weight_kg": 7.3,
    "zone_type": "ambient",
    "shift": "Morning",
}

# A single TestClient instance – requests are issued inside patch contexts
_CLIENT = TestClient(app)


def _mock_model(return_value: float = 342.0) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = [return_value]
    return m


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_http_200(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.get("/health")
        assert resp.status_code == 200

    def test_body_status_ok(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /predict – happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_returns_http_200_with_model(self):
        with patch.object(_main, "_MODEL", _mock_model(342.0)):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_prediction_key(self):
        with patch.object(_main, "_MODEL", _mock_model(342.0)):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert "prediction" in resp.json()

    def test_prediction_is_numeric(self):
        with patch.object(_main, "_MODEL", _mock_model(342.0)):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(resp.json()["prediction"], (int, float))

    def test_prediction_matches_model_output(self):
        with patch.object(_main, "_MODEL", _mock_model(99.5)):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert pytest.approx(resp.json()["prediction"], abs=1e-3) == 99.5

    def test_response_has_only_prediction_key(self):
        with patch.object(_main, "_MODEL", _mock_model(1.0)):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert list(resp.json().keys()) == ["prediction"]

    def test_boundary_aisle_distance_exactly_700_accepted(self):
        payload = {**VALID_PAYLOAD, "aisle_distance_m": 700.0}
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_boundary_shelf_height_exactly_0_accepted(self):
        payload = {**VALID_PAYLOAD, "shelf_height_m": 0.0}
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict – HTTP 422 validation errors
# ---------------------------------------------------------------------------

class TestPredictValidation422:
    def test_aisle_distance_750_returns_422(self):
        """aisle_distance_m = 750 must be rejected with HTTP 422."""
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "aisle_distance_m": 750})
        assert resp.status_code == 422

    def test_shelf_height_negative_1_5_returns_422(self):
        """shelf_height_m = -1.5 must be rejected with HTTP 422."""
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "shelf_height_m": -1.5})
        assert resp.status_code == 422

    def test_aisle_distance_700_01_returns_422(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "aisle_distance_m": 700.01})
        assert resp.status_code == 422

    def test_shelf_height_very_negative_returns_422(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "shelf_height_m": -100.0})
        assert resp.status_code == 422

    def test_missing_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "zone_type"}
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={})
        assert resp.status_code == 422

    def test_wrong_type_returns_422(self):
        with patch.object(_main, "_MODEL", _mock_model()):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "items_in_order": "five"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict – HTTP 503 when model is unavailable
# ---------------------------------------------------------------------------

class TestPredictModelUnavailable:
    def test_returns_http_503_when_model_is_none(self):
        """With no model loaded the endpoint must return HTTP 503."""
        with patch.object(_main, "_MODEL", None):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_503_response_has_detail_key(self):
        with patch.object(_main, "_MODEL", None):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert "detail" in resp.json(), f"Expected 'detail' key, got: {resp.json()}"

    def test_503_detail_is_non_empty_string(self):
        with patch.object(_main, "_MODEL", None):
            resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(resp.json()["detail"], str)
        assert len(resp.json()["detail"]) > 0

    def test_422_still_raised_before_503_check(self):
        """Validation errors (422) must fire even when model is None."""
        with patch.object(_main, "_MODEL", None):
            resp = _CLIENT.post("/predict", json={**VALID_PAYLOAD, "aisle_distance_m": 750})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic schema – field-level unit tests (no event loop required)
# ---------------------------------------------------------------------------

class TestPredictRequestSchema:
    def test_valid_payload_accepted(self):
        req = PredictRequest(**VALID_PAYLOAD)
        assert req.items_in_order == 5
        assert req.aisle_distance_m == 120.5
        assert req.zone_type == "ambient"
        assert req.shift == "Morning"

    def test_all_eight_fields_accessible(self):
        req = PredictRequest(**VALID_PAYLOAD)
        for field in _main._FEATURE_COLS:
            assert hasattr(req, field), f"Missing field: {field}"

    @pytest.mark.parametrize("missing_field", [
        "items_in_order", "aisle_distance_m", "picker_exp_months",
        "shelf_height_m", "shift_hour", "order_weight_kg", "zone_type", "shift",
    ])
    def test_each_field_is_required(self, missing_field):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
        with pytest.raises(ValidationError):
            PredictRequest(**payload)

    def test_aisle_distance_above_700_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(**{**VALID_PAYLOAD, "aisle_distance_m": 701.0})
        errors = exc_info.value.errors()
        aisle_errors = [e for e in errors if e["loc"][0] == "aisle_distance_m"]
        assert aisle_errors and aisle_errors[0]["type"] == "less_than_equal"

    def test_shelf_height_below_zero_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(**{**VALID_PAYLOAD, "shelf_height_m": -1.0})
        errors = exc_info.value.errors()
        shelf_errors = [e for e in errors if e["loc"][0] == "shelf_height_m"]
        assert shelf_errors and shelf_errors[0]["type"] == "greater_than_equal"


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class TestStructuredLogging:
    def _capture_predict_logs(self, model: object = None) -> list[dict]:
        """Call POST /predict via TestClient and collect JSON lines written to stdout.

        Does not swallow any unexpected exceptions during prediction execution.
        """
        buf = StringIO()
        logger = logging.getLogger("mp153.api")
        original_streams: list[tuple] = []
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                original_streams.append((h, h.stream))
                h.stream = buf

        try:
            with patch.object(_main, "_MODEL", model):
                resp = _CLIENT.post("/predict", json=VALID_PAYLOAD)
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        finally:
            for h, orig in original_streams:
                h.stream = orig

        result = []
        for line in buf.getvalue().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return result

    def test_exactly_one_json_line_per_successful_prediction(self):
        """Verify that exactly one JSON log line is emitted for each successful prediction."""
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        assert len(logs) == 1, f"Expected exactly 1 JSON log line, got {len(logs)}: {logs}"

    def test_one_json_line_per_successful_prediction(self):
        """Preserve backwards-compatible test name verifying exactly 1 JSON log line."""
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        assert len(logs) == 1

    def test_log_line_has_prediction_key(self):
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        assert any("prediction" in log for log in logs)

    def test_log_line_prediction_is_numeric(self):
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        for log in logs:
            if "prediction" in log:
                assert isinstance(log["prediction"], (int, float))

    def test_log_line_has_no_record_id_pii(self):
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        for log in logs:
            assert "record_id" not in log

    def test_log_line_has_level_field(self):
        logs = self._capture_predict_logs(model=_mock_model(100.0))
        for log in logs:
            assert "level" in log

    def test_no_file_handler_on_logger(self):
        logger = logging.getLogger("mp153.api")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert not file_handlers, "Logger must not have any FileHandlers"

    def test_logger_handler_targets_stdout(self):
        logger = logging.getLogger("mp153.api")
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                assert h.stream is sys.stdout or \
                    getattr(h.stream, "name", None) in ("<stdout>", "stdout")
