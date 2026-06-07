"""Tests for MODULE_008_BENCHMARK — extracted from tests/test_gpts_core.py::TestBenchmark."""
from __future__ import annotations

import pytest

from gpts_core.benchmark import (
    train_linear,
    predict_linear,
    mse_score,
    compare_prediction_lock,
)


class TestBenchmark:
    def test_train_linear_perfect_fit(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        w, b = train_linear(X, y, epochs=10000, lr=0.05)
        preds = predict_linear(X, w, b, clip=False)
        for pred, true in zip(preds, y):
            assert abs(pred - true) < 0.5, f"pred={pred:.3f} true={true}"

    def test_predict_linear_clip(self):
        preds = predict_linear([[100.0]], [1.0], 0.0, clip=True)
        assert preds[0] == 1.0

    def test_mse_score_perfect(self):
        preds = {"a": 0.8, "b": 0.6}
        truth = {"a": 0.8, "b": 0.6}
        score = mse_score(preds, truth)
        assert abs(score - 1.0) < 0.01

    def test_mse_score_missing_penalty(self):
        preds = {}
        truth = {"a": 0.8}
        score = mse_score(preds, truth, missing_penalty=0.1)
        assert score < 1.0

    def test_mse_score_range(self):
        preds = {"a": 0.3}
        truth = {"a": 0.9}
        score = mse_score(preds, truth)
        assert 0.0 <= score <= 1.0

    def test_compare_prediction_lock_strong_pass(self):
        lock = {"gate_g2_numeric_prediction": {
            "primary_metric": "psiomega_mean",
            "target_next_shadow_phase": 13,
            "predicted_value": 0.56,
            "strong_pass_interval": [0.52, 0.62],
            "normal_pass_interval": [0.48, 0.66],
        }}
        obs = {
            "run_id": "r1", "phase": 13, "psiomega_mean": 0.57,
            "ledger_replay_status": "PASS", "control_effects": 0,
            "production_unlocked": False, "stdout_retained": True, "stderr_retained": True,
        }
        result = compare_prediction_lock(lock, obs)
        assert result["verdict"] == "G2_STRONG_PASS_LAB_ONLY"
        assert result["production_unlocked"] is False

    def test_compare_prediction_lock_out_of_bounds(self):
        lock = {"gate_g2_numeric_prediction": {
            "primary_metric": "psiomega_mean",
            "target_next_shadow_phase": 13,
            "predicted_value": 0.56,
            "strong_pass_interval": [0.52, 0.62],
            "normal_pass_interval": [0.48, 0.66],
        }}
        obs = {
            "run_id": "r1", "phase": 13, "psiomega_mean": 0.90,
            "ledger_replay_status": "PASS", "control_effects": 0,
            "production_unlocked": False, "stdout_retained": True, "stderr_retained": True,
        }
        result = compare_prediction_lock(lock, obs)
        assert result["verdict"] == "G2_FAIL_RECALIBRATION_REQUIRED"

    def test_compare_prediction_lock_fail_closed_on_errors(self):
        lock = {"gate_g2_numeric_prediction": {"primary_metric": "x", "target_next_shadow_phase": 1}}
        result = compare_prediction_lock(lock, {"phase": 99})
        assert "FAIL" in result["verdict"]
