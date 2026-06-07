"""MODULE_008_BENCHMARK — runnable usage examples.

Demonstrates: minmax_fit, minmax_transform, train_linear, predict_linear,
mse_score, and compare_prediction_lock.
"""
from __future__ import annotations

from gpts_core.benchmark import (
    minmax_fit,
    minmax_transform,
    train_linear,
    predict_linear,
    mse_score,
    compare_prediction_lock,
)

# ---------------------------------------------------------------------------
# Example 1: Train a linear model via gradient descent (no external deps)
# ---------------------------------------------------------------------------

# Simple y = 2*x relationship
X_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
y_train = [2.0, 4.0, 6.0, 8.0, 10.0]

# Normalize features before training
mins, maxs = minmax_fit(X_train)
X_scaled = minmax_transform(X_train, mins, maxs)

weights, bias = train_linear(X_scaled, y_train, epochs=5000, lr=0.05)
preds = predict_linear(X_scaled, weights, bias, clip=False)

print("[Example 1] Linear regression (y = 2x)")
for xi, yi, pi in zip(X_train, y_train, preds):
    print(f"  x={xi[0]:.1f}  true={yi:.1f}  pred={pi:.3f}  err={abs(pi - yi):.4f}")

# ---------------------------------------------------------------------------
# Example 2: MSE stability scoring with penalties
# ---------------------------------------------------------------------------

# Perfect predictions
predictions_perfect = {"sample_01": 0.82, "sample_02": 0.55, "sample_03": 0.70}
truth = {"sample_01": 0.82, "sample_02": 0.55, "sample_03": 0.70}
score_perfect = mse_score(predictions_perfect, truth)
print(f"\n[Example 2] Perfect predictions score: {score_perfect:.4f}")  # ~1.0

# Some missing predictions (penalty applied)
predictions_partial = {"sample_01": 0.82}
score_partial = mse_score(predictions_partial, truth, missing_penalty=0.10)
print(f"[Example 2] Partial predictions score: {score_partial:.4f}")  # < 1.0

# Out-of-range predictions (range penalty applied)
predictions_oob = {"sample_01": 1.5, "sample_02": -0.2, "sample_03": 0.70}
score_oob = mse_score(predictions_oob, truth, range_penalty=0.05)
print(f"[Example 2] Out-of-range predictions score: {score_oob:.4f}")

# ---------------------------------------------------------------------------
# Example 3: Prediction lock comparison for shadow observation gate G2
# ---------------------------------------------------------------------------

prediction_lock = {
    "gate_g2_numeric_prediction": {
        "primary_metric": "psiomega_mean",
        "target_next_shadow_phase": 13,
        "predicted_value": 0.56,
        "strong_pass_interval": [0.52, 0.62],
        "normal_pass_interval": [0.48, 0.66],
    }
}

# Observation within strong pass interval
obs_strong = {
    "run_id": "run_013",
    "phase": 13,
    "psiomega_mean": 0.57,
    "ledger_replay_status": "PASS",
    "control_effects": 0,
    "production_unlocked": False,
    "stdout_retained": True,
    "stderr_retained": True,
}
result_strong = compare_prediction_lock(prediction_lock, obs_strong)
print(f"\n[Example 3] Strong pass verdict: {result_strong['verdict']}")
print(f"[Example 3] production_unlocked: {result_strong['production_unlocked']}")
print(f"[Example 3] delta from predicted: {result_strong['delta']:.4f}")

# Observation outside all intervals
obs_fail = dict(obs_strong, psiomega_mean=0.99)
result_fail = compare_prediction_lock(prediction_lock, obs_fail)
print(f"[Example 3] Out-of-bounds verdict: {result_fail['verdict']}")
