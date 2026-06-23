"""Tests for gpts_core.benchmark — linear model and scoring utilities."""
from __future__ import annotations

import math
import pytest

import csv

from gpts_core.benchmark import (
    _to_float,
    _dot,
    minmax_fit,
    minmax_transform,
    train_linear,
    predict_linear,
    mse_score,
    compare_prediction_lock,
    read_feature_csv,
    read_score_csv,
)


# ---------------------------------------------------------------------------
# _to_float
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_bool_true(self):
        assert _to_float(True) == 1.0

    def test_bool_false(self):
        assert _to_float(False) == 0.0

    def test_string_true(self):
        assert _to_float("true") == 1.0

    def test_string_yes(self):
        assert _to_float("yes") == 1.0

    def test_string_false(self):
        assert _to_float("false") == 0.0

    def test_string_no(self):
        assert _to_float("no") == 0.0

    def test_numeric_string(self):
        assert _to_float("3.14") == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# _dot
# ---------------------------------------------------------------------------

class TestDot:
    def test_basic(self):
        assert _dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_zero_vectors(self):
        assert _dot([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# minmax_fit and minmax_transform
# ---------------------------------------------------------------------------

class TestMinmax:
    def test_fit_returns_mins_maxs(self):
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        mins, maxs = minmax_fit(matrix)
        assert mins == [1.0, 2.0]
        assert maxs == [5.0, 6.0]

    def test_transform_scales_to_zero_one(self):
        matrix = [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]
        mins, maxs = minmax_fit(matrix)
        scaled = minmax_transform(matrix, mins, maxs)
        assert scaled[0] == [0.0, 0.0]
        assert scaled[-1] == [1.0, 1.0]

    def test_zero_variance_feature_becomes_zero(self):
        matrix = [[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]]
        mins, maxs = minmax_fit(matrix)
        scaled = minmax_transform(matrix, mins, maxs)
        assert all(row[0] == 0.0 for row in scaled)


# ---------------------------------------------------------------------------
# train_linear + predict_linear
# ---------------------------------------------------------------------------

class TestLinearModel:
    def test_constant_target_converges(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [3.0, 3.0, 3.0, 3.0, 3.0]
        w, b = train_linear(X, y, epochs=200)
        preds = predict_linear(X, w, b, clip=False)
        for p in preds:
            assert abs(p - 3.0) < 0.5

    def test_linear_relationship(self):
        X = [[float(i)] for i in range(10)]
        y = [float(i) * 0.1 for i in range(10)]
        mins, maxs = minmax_fit(X)
        X_norm = minmax_transform(X, mins, maxs)
        y_norm = [v for v in y]
        w, b = train_linear(X_norm, y_norm, epochs=3000, lr=0.1)
        preds = predict_linear(X_norm, w, b, clip=False)
        mse = sum((p - t) ** 2 for p, t in zip(preds, y_norm)) / len(y_norm)
        assert mse < 0.01

    def test_predict_clips_by_default(self):
        w, b = [10.0], 100.0
        preds = predict_linear([[1.0], [2.0]], w, b, clip=True)
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_predict_no_clip(self):
        w, b = [10.0], 100.0
        preds = predict_linear([[1.0]], w, b, clip=False)
        assert preds[0] > 1.0

    def test_empty_X_returns_empty_weights(self):
        w, b = train_linear([], [], epochs=10)
        assert w == []

    def test_empty_X_predict_empty(self):
        preds = predict_linear([], [], 0.0)
        assert preds == []


# ---------------------------------------------------------------------------
# read_feature_csv / read_score_csv
# ---------------------------------------------------------------------------

class TestCsvIO:
    def test_read_feature_csv(self, tmp_path):
        p = tmp_path / "features.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["a", "b"])
            w.writeheader()
            w.writerow({"a": "1.0", "b": "2.0"})
            w.writerow({"a": "3.0", "b": "4.0"})
        result = read_feature_csv(p, ["a", "b"])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_read_score_csv(self, tmp_path):
        p = tmp_path / "scores.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "stability_score"])
            w.writeheader()
            w.writerow({"sample_id": "s1", "stability_score": "0.8"})
            w.writerow({"sample_id": "s2", "stability_score": "0.9"})
        result = read_score_csv(p)
        assert result["s1"] == pytest.approx(0.8)
        assert result["s2"] == pytest.approx(0.9)

    def test_read_score_csv_invalid_value(self, tmp_path):
        p = tmp_path / "scores.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "stability_score"])
            w.writeheader()
            w.writerow({"sample_id": "s1", "stability_score": "not_a_number"})
        result = read_score_csv(p)
        import math
        assert math.isnan(result["s1"])


# ---------------------------------------------------------------------------
# mse_score
# ---------------------------------------------------------------------------

class TestMseScore:
    def test_perfect_predictions(self):
        truth = {"a": 0.5, "b": 0.8}
        preds = {"a": 0.5, "b": 0.8}
        score = mse_score(preds, truth)
        assert score == pytest.approx(1.0)

    def test_missing_prediction_penalized(self):
        truth = {"a": 0.5, "b": 0.5}
        preds = {"a": 0.5}
        score = mse_score(preds, truth, missing_penalty=0.10)
        assert score < 1.0

    def test_invalid_prediction_penalized(self):
        truth = {"a": 0.5}
        preds = {"a": float("inf")}
        score = mse_score(preds, truth, invalid_penalty=0.20)
        assert score < 1.0

    def test_out_of_range_penalized(self):
        truth = {"a": 0.5}
        preds = {"a": 1.5}
        score = mse_score(preds, truth, range_penalty=0.05)
        assert score < 1.0

    def test_empty_truth_returns_zero(self):
        score = mse_score({}, {})
        assert score == 0.0

    def test_score_bounded_zero_one(self):
        truth = {"a": 0.0, "b": 1.0}
        preds = {"a": 0.9, "b": 0.1}
        score = mse_score(preds, truth)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compare_prediction_lock
# ---------------------------------------------------------------------------

def _valid_prediction_lock(predicted=0.56, strong=(0.52, 0.62), normal=(0.48, 0.66)):
    return {
        "gate_g2_numeric_prediction": {
            "primary_metric": "psiomega_mean",
            "target_next_shadow_phase": 13,
            "predicted_value": predicted,
            "strong_pass_interval": list(strong),
            "normal_pass_interval": list(normal),
        }
    }


def _valid_observation(metric_value=0.55, phase=13):
    return {
        "run_id": "run_001",
        "phase": phase,
        "psiomega_mean": metric_value,
        "ledger_replay_status": "PASS",
        "control_effects": 0,
        "production_unlocked": False,
        "stdout_retained": True,
        "stderr_retained": True,
    }


class TestComparePredictionLock:
    def test_strong_pass(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(metric_value=0.57),
        )
        assert result["verdict"] == "G2_STRONG_PASS_LAB_ONLY"
        assert result["production_unlocked"] is False

    def test_normal_pass(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(metric_value=0.50),
        )
        assert result["verdict"] == "G2_NORMAL_PASS_LAB_ONLY"

    def test_out_of_bounds(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(metric_value=0.90),
        )
        assert result["verdict"] == "G2_FAIL_RECALIBRATION_REQUIRED"

    def test_missing_phase_is_error(self):
        obs = _valid_observation()
        del obs["phase"]
        result = compare_prediction_lock(_valid_prediction_lock(), obs)
        assert result["verdict"] == "FAIL_CLOSED_INVALID_INPUT"
        assert "MISSING_PHASE" in result["input_errors"]

    def test_wrong_phase_is_error(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(phase=99),
        )
        assert result["verdict"] == "FAIL_CLOSED_INVALID_INPUT"
        assert any("PHASE_MISMATCH" in e for e in result["input_errors"])

    def test_ledger_replay_not_pass_is_error(self):
        obs = _valid_observation()
        obs["ledger_replay_status"] = "FAIL"
        result = compare_prediction_lock(_valid_prediction_lock(), obs)
        assert "LEDGER_REPLAY_NOT_PASS" in result["input_errors"]

    def test_result_has_hash(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(metric_value=0.57),
        )
        assert "result_hash" in result
        assert result["result_hash"].startswith("sha256:")

    def test_production_always_unlocked_false(self):
        result = compare_prediction_lock(
            _valid_prediction_lock(),
            _valid_observation(),
        )
        assert result["production_unlocked"] is False

    def test_empty_lock_has_errors(self):
        result = compare_prediction_lock({}, _valid_observation())
        assert "verdict" in result
        assert result["production_unlocked"] is False
