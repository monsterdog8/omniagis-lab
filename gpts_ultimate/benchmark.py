"""Benchmark utilities: stdlib-only linear baseline model and MSE scoring.

Includes:
  - Linear regression via gradient descent (no external deps)
  - Weighted stability scoring with missing/invalid penalties
  - Pre-registered prediction lock comparison for shadow observation gates
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Linear model (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return 1.0
    if s in {"false", "0", "no"}:
        return 0.0
    return float(s)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def minmax_fit(matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Fit min-max scaler from a feature matrix."""
    cols = list(zip(*matrix))
    return [min(c) for c in cols], [max(c) for c in cols]


def minmax_transform(
    matrix: List[List[float]],
    mins: List[float],
    maxs: List[float],
) -> List[List[float]]:
    """Apply min-max scaling."""
    result = []
    for row in matrix:
        out = []
        for v, lo, hi in zip(row, mins, maxs):
            denom = hi - lo
            out.append(0.0 if denom == 0 else (v - lo) / denom)
        result.append(out)
    return result


def train_linear(
    X: List[List[float]],
    y: List[float],
    epochs: int = 5000,
    lr: float = 0.05,
) -> Tuple[List[float], float]:
    """Train a linear regression model via gradient descent.

    Returns (weights, bias).
    """
    n = len(X)
    m = len(X[0]) if X else 0
    weights = [0.0] * m
    bias = sum(y) / max(1, len(y))
    for _ in range(epochs):
        grad_w = [0.0] * m
        grad_b = 0.0
        for xi, yi in zip(X, y):
            err = _dot(weights, xi) + bias - yi
            for j in range(m):
                grad_w[j] += (2.0 / n) * err * xi[j]
            grad_b += (2.0 / n) * err
        weights = [w - lr * g for w, g in zip(weights, grad_w)]
        bias -= lr * grad_b
    return weights, bias


def predict_linear(
    X: List[List[float]],
    weights: List[float],
    bias: float,
    clip: bool = True,
) -> List[float]:
    """Predict with a linear model, optionally clipped to [0, 1]."""
    preds = [_dot(weights, xi) + bias for xi in X]
    if clip:
        return [min(1.0, max(0.0, p)) for p in preds]
    return preds


def read_feature_csv(path: Path, feature_cols: List[str]) -> List[List[float]]:
    """Read feature matrix from CSV."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append([_to_float(row[c]) for c in feature_cols])
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def mse_score(
    predictions: Dict[str, float],
    truth: Dict[str, float],
    missing_penalty: float = 0.10,
    invalid_penalty: float = 0.20,
    range_penalty: float = 0.05,
) -> float:
    """Weighted stability score: 1 - MSE, minus per-sample penalties.

    Returns score in [0, 1].
    """
    squared_errors = []
    total_penalty = 0.0
    for sid, true_val in truth.items():
        if sid not in predictions:
            total_penalty += missing_penalty
            squared_errors.append(1.0)
            continue
        pred = predictions[sid]
        if not math.isfinite(pred):
            total_penalty += invalid_penalty
            pred = 0.0
        if pred < 0.0 or pred > 1.0:
            total_penalty += range_penalty
            pred = min(1.0, max(0.0, pred))
        squared_errors.append((pred - true_val) ** 2)
    if not squared_errors:
        return 0.0
    base = max(0.0, 1.0 - sum(squared_errors) / len(squared_errors))
    return max(0.0, min(1.0, base - total_penalty))


def read_score_csv(path: Path) -> Dict[str, float]:
    """Read sample_id → stability_score from CSV."""
    result: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                result[row["sample_id"]] = float(row["stability_score"])
            except (ValueError, KeyError):
                result[row.get("sample_id", "UNKNOWN")] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Prediction lock comparator (shadow observation gate G2)
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compare_prediction_lock(
    prediction_lock: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a shadow observation against a pre-registered numeric prediction lock.

    Returns a structured verdict dict. Production is never unlocked automatically.
    """
    g2 = prediction_lock.get("gate_g2_numeric_prediction", {})
    metric = g2.get("primary_metric", "psiomega_mean")
    target_phase = int(g2.get("target_next_shadow_phase", -1))
    predicted_val = g2.get("predicted_value")
    strong = g2.get("strong_pass_interval")
    normal = g2.get("normal_pass_interval")

    errors: List[str] = []
    phase = observation.get("phase")
    measured = observation.get(metric)

    if phase is None:
        errors.append("MISSING_PHASE")
    elif int(phase) != target_phase:
        errors.append(f"PHASE_MISMATCH:expected={target_phase},observed={phase}")
    if measured is None:
        errors.append(f"MISSING_METRIC:{metric}")
    if observation.get("ledger_replay_status") != "PASS":
        errors.append("LEDGER_REPLAY_NOT_PASS")
    if int(observation.get("control_effects", -1)) != 0:
        errors.append("CONTROL_EFFECTS_NOT_ZERO")
    if observation.get("production_unlocked") is not False:
        errors.append("PRODUCTION_UNLOCKED_NOT_FALSE")
    if observation.get("stdout_retained") is not True:
        errors.append("STDOUT_NOT_RETAINED")
    if observation.get("stderr_retained") is not True:
        errors.append("STDERR_NOT_RETAINED")

    interval_result = "NOT_EVALUATED"
    delta: Optional[float] = None
    if measured is not None and predicted_val is not None and strong and normal:
        m = float(measured)
        delta = m - float(predicted_val)
        if float(strong[0]) <= m <= float(strong[1]):
            interval_result = "STRONG_PASS"
        elif float(normal[0]) <= m <= float(normal[1]):
            interval_result = "NORMAL_PASS"
        else:
            interval_result = "OUT_OF_BOUNDS"

    if errors:
        verdict = "FAIL_CLOSED_INVALID_INPUT"
    elif interval_result == "STRONG_PASS":
        verdict = "G2_STRONG_PASS_LAB_ONLY"
    elif interval_result == "NORMAL_PASS":
        verdict = "G2_NORMAL_PASS_LAB_ONLY"
    elif interval_result == "OUT_OF_BOUNDS":
        verdict = "G2_FAIL_RECALIBRATION_REQUIRED"
    else:
        verdict = "FAIL_CLOSED_NOT_EVALUATED"

    result: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "primary_metric": metric,
        "target_phase": target_phase,
        "predicted_value": predicted_val,
        "measured_value": measured,
        "delta": delta,
        "strong_pass_interval": strong,
        "normal_pass_interval": normal,
        "interval_result": interval_result,
        "input_errors": errors,
        "verdict": verdict,
        "production_unlocked": False,
        "production_impact": (
            "MAY_REQUEST_REVIEW_ONLY" if verdict == "G2_STRONG_PASS_LAB_ONLY"
            else "NO_PROMOTION"
        ),
    }
    result["result_hash"] = _sha256_json(result)
    return result
