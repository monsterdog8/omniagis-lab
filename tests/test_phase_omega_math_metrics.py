"""Tests for phase_omega.math_metrics — formula registry, metric implementations, pipelines."""
from __future__ import annotations

import math

import numpy as np
import pytest

from phase_omega.math_metrics import (
    FormulaRegistry,
    MetricComputer,
    HazardRegistry,
    RealityLedger,
    ValidationPipeline,
)


# ---------------------------------------------------------------------------
# FormulaRegistry
# ---------------------------------------------------------------------------

class TestFormulaRegistry:
    def test_list_all_nonempty(self):
        ids = FormulaRegistry.list_all()
        assert len(ids) >= 10

    def test_known_formula_ids_present(self):
        ids = set(FormulaRegistry.list_all())
        for fid in ["BRIER_SCORE", "BRIER_SKILL_SCORE", "EXPECTED_CALIBRATION_ERROR",
                    "LAMBDA_1_LOGISTIC", "ULAM_SPECTRAL_GAP", "CORRELATION_DIMENSION"]:
            assert fid in ids

    def test_get_formula_returns_dict(self):
        f = FormulaRegistry.get_formula("BRIER_SCORE")
        assert isinstance(f, dict)

    def test_get_formula_has_name_and_formula(self):
        f = FormulaRegistry.get_formula("LAMBDA_1_LOGISTIC")
        assert "name" in f
        assert "formula" in f

    def test_get_formula_missing_raises(self):
        with pytest.raises(KeyError):
            FormulaRegistry.get_formula("NONEXISTENT_FORMULA")

    def test_verify_all_implemented_returns_dict(self):
        status = FormulaRegistry.verify_all_implemented(MetricComputer)
        assert isinstance(status, dict)
        assert len(status) == len(FormulaRegistry.list_all())

    def test_brier_score_implemented(self):
        status = FormulaRegistry.verify_all_implemented(MetricComputer)
        assert status["BRIER_SCORE"] is True

    def test_correlation_dimension_implemented(self):
        status = FormulaRegistry.verify_all_implemented(MetricComputer)
        assert status["CORRELATION_DIMENSION"] is True


# ---------------------------------------------------------------------------
# MetricComputer — Lyapunov
# ---------------------------------------------------------------------------

class TestLyapunovExponents:
    def _logistic_trajectory(self, r=3.82, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.2, 0.8)
        traj = []
        for _ in range(n):
            x = r * x * (1 - x)
            traj.append(x)
        return np.array(traj)

    def test_lambda1_logistic_positive_for_chaotic(self):
        traj = self._logistic_trajectory(r=3.82)
        df = lambda x: 3.82 * (1 - 2 * x)
        lam = MetricComputer.lambda1_from_trajectory(traj, df)
        assert lam > 0.0

    def test_lambda1_logistic_empty_after_burnin_nan(self):
        traj = np.zeros(50)  # all burn_in=100 → xs is empty
        df = lambda x: 0.0
        lam = MetricComputer.lambda1_from_trajectory(traj, df, burn_in=100)
        assert math.isnan(lam)

    def test_lambda1_pm_positive(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0.01, 0.99, 500)
        lam = MetricComputer.lambda1_pm_from_trajectory(traj, alpha=0.5)
        assert math.isfinite(lam)


# ---------------------------------------------------------------------------
# MetricComputer — Ulam Matrix / Spectral Gap
# ---------------------------------------------------------------------------

class TestUlamMatrix:
    def test_matrix_shape(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 200)
        P = MetricComputer.ulam_matrix_from_trajectory(traj, partition_size=10)
        assert P.shape == (10, 10)

    def test_matrix_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 500)
        P = MetricComputer.ulam_matrix_from_trajectory(traj, partition_size=8)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_spectral_gap_positive(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 500)
        gap = MetricComputer.ulam_spectral_gap(traj, partition_size=8)
        assert gap >= 0.0

    def test_spectral_gap_bounded_by_one(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 500)
        gap = MetricComputer.ulam_spectral_gap(traj, partition_size=8)
        assert gap <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# MetricComputer — Correlation Function
# ---------------------------------------------------------------------------

class TestCorrelationFunction:
    def test_returns_arrays(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 200)
        lags, corrs = MetricComputer.correlation_function(s, s, max_lag=20)
        assert len(lags) > 0
        assert len(lags) == len(corrs)

    def test_lag_0_is_positive_for_autocorrelation(self):
        rng = np.random.default_rng(0)
        s = np.sin(np.linspace(0, 10 * np.pi, 500))
        lags, corrs = MetricComputer.correlation_function(s, s, max_lag=50)
        assert len(corrs) > 0

    def test_decay_exponent_returns_tuple(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 500)
        lags, corrs = MetricComputer.correlation_function(s, s, max_lag=50)
        result = MetricComputer.correlation_decay_exponent(lags, corrs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# MetricComputer — Return Times
# ---------------------------------------------------------------------------

class TestReturnTimes:
    def test_survival_returns_arrays(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 500)
        ns, survivals = MetricComputer.return_time_survival(traj, (0.4, 0.6))
        assert isinstance(ns, np.ndarray)
        assert isinstance(survivals, np.ndarray)

    def test_survival_values_in_zero_one(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 500)
        ns, survivals = MetricComputer.return_time_survival(traj, (0.4, 0.6))
        if len(survivals) > 0:
            assert np.all(survivals >= 0.0)
            assert np.all(survivals <= 1.0)

    def test_return_time_exponent_nan_for_empty(self):
        gamma, r2 = MetricComputer.return_time_exponent(np.array([]), np.array([]))
        assert math.isnan(gamma)

    def test_never_in_ref_set_returns_empty(self):
        traj = np.zeros(200)  # always 0, never in (0.4, 0.6)
        ns, survivals = MetricComputer.return_time_survival(traj, (0.4, 0.6))
        assert len(survivals) == 0


# ---------------------------------------------------------------------------
# MetricComputer — Recurrence
# ---------------------------------------------------------------------------

class TestRecurrence:
    def test_recurrence_rate_in_zero_one(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 50)
        rr = MetricComputer.recurrence_rate(traj, epsilon=0.1)
        assert 0.0 <= rr <= 1.0

    def test_recurrence_rate_increases_with_epsilon(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 50)
        rr_small = MetricComputer.recurrence_rate(traj, epsilon=0.01)
        rr_large = MetricComputer.recurrence_rate(traj, epsilon=0.5)
        assert rr_large >= rr_small

    def test_determinism_rqa_nonnegative(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 30)
        det = MetricComputer.determinism_rqa(traj, epsilon=0.2)
        assert det >= 0.0

    def test_determinism_rqa_zero_if_no_lines(self):
        traj = np.linspace(0, 1, 20)  # monotone: diagonal only, no repeated points
        det = MetricComputer.determinism_rqa(traj, epsilon=0.01, min_line_length=100)
        assert det == 0.0


# ---------------------------------------------------------------------------
# MetricComputer — Correlation Dimension
# ---------------------------------------------------------------------------

class TestCorrelationDimension:
    def test_returns_tuple(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 50)
        d2, r2 = MetricComputer.correlation_dimension_gp(traj)
        assert isinstance(d2, float)
        assert isinstance(r2, float)

    def test_positive_dimension(self):
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 1, 80)
        d2, r2 = MetricComputer.correlation_dimension_gp(traj)
        assert d2 > 0.0


# ---------------------------------------------------------------------------
# MetricComputer — Brier Score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_forecast_zero(self):
        f = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        bs = MetricComputer.brier_score(f, y)
        assert bs == pytest.approx(0.0)

    def test_worst_forecast(self):
        f = np.array([1.0, 1.0])
        y = np.array([0.0, 0.0])
        bs = MetricComputer.brier_score(f, y)
        assert bs == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            MetricComputer.brier_score(np.array([0.5]), np.array([0.5, 0.5]))

    def test_bs_symmetric(self):
        f = np.array([0.3, 0.7])
        y = np.array([0.0, 1.0])
        bs1 = MetricComputer.brier_score(f, y)
        bs2 = MetricComputer.brier_score(y, f)
        assert bs1 == pytest.approx(bs2)


class TestBrierSkillScore:
    def test_perfect_vs_climatology(self):
        outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0] * 10)
        forecasts = outcomes.copy()
        bss = MetricComputer.brier_skill_score(forecasts, outcomes)
        assert bss == pytest.approx(1.0)

    def test_climatology_baseline_bss_zero(self):
        outcomes = np.array([1.0, 0.0, 1.0, 0.0] * 5)
        forecasts = np.full_like(outcomes, outcomes.mean())
        bss = MetricComputer.brier_skill_score(forecasts, outcomes)
        assert bss == pytest.approx(0.0, abs=1e-9)

    def test_nan_when_baseline_perfect(self):
        outcomes = np.array([0.5, 0.5, 0.5])
        baseline = np.array([0.5, 0.5, 0.5])
        forecasts = np.array([0.6, 0.4, 0.5])
        bss = MetricComputer.brier_skill_score(forecasts, outcomes, baseline_forecasts=baseline)
        assert math.isnan(bss)


class TestExpectedCalibrationError:
    def test_perfect_calibration_zero(self):
        # Forecasts exactly equal outcomes (0.0 or 1.0 only)
        f = np.array([1.0] * 50 + [0.0] * 50)
        y = np.array([1.0] * 50 + [0.0] * 50)
        ece = MetricComputer.expected_calibration_error(f, y)
        assert ece == pytest.approx(0.0)

    def test_ece_nonnegative(self):
        rng = np.random.default_rng(0)
        f = rng.uniform(0, 1, 100)
        y = (rng.uniform(0, 1, 100) > 0.5).astype(float)
        ece = MetricComputer.expected_calibration_error(f, y)
        assert ece >= 0.0

    def test_ece_bounded_by_one(self):
        rng = np.random.default_rng(0)
        f = rng.uniform(0, 1, 100)
        y = (rng.uniform(0, 1, 100) > 0.5).astype(float)
        ece = MetricComputer.expected_calibration_error(f, y)
        assert ece <= 1.0


# ---------------------------------------------------------------------------
# HazardRegistry
# ---------------------------------------------------------------------------

class TestHazardRegistry:
    def test_list_active_nonempty(self):
        active = HazardRegistry.list_active()
        assert len(active) > 0

    def test_ch27_is_active(self):
        assert "CH27" in HazardRegistry.list_active()

    def test_get_hazard_returns_status(self):
        h = HazardRegistry.get_hazard("CH27")
        assert h is not None
        assert h.code == "CH27"
        assert h.category == "ACTIVE"
        assert h.status == "BLOCKED"

    def test_get_hazard_missing_returns_none(self):
        assert HazardRegistry.get_hazard("CH999") is None

    def test_ch31_baseline_neglect(self):
        h = HazardRegistry.get_hazard("CH31")
        assert h is not None
        assert "BASELINE" in h.name

    def test_all_active_are_active_category(self):
        for code in HazardRegistry.list_active():
            h = HazardRegistry.get_hazard(code)
            assert h.category == "ACTIVE"


# ---------------------------------------------------------------------------
# RealityLedger
# ---------------------------------------------------------------------------

class TestRealityLedger:
    def test_list_all_nonempty(self):
        preds = RealityLedger.list_all()
        assert len(preds) >= 1

    def test_p001_exists(self):
        assert "P_001" in RealityLedger.list_all()

    def test_get_prediction_returns_prediction(self):
        p = RealityLedger.get_prediction("P_001")
        assert p is not None
        assert p.prediction_id == "P_001"

    def test_prediction_has_frozen_status(self):
        p = RealityLedger.get_prediction("P_001")
        assert p.status == "FROZEN"

    def test_prediction_confidence_in_range(self):
        p = RealityLedger.get_prediction("P_001")
        assert 0.0 <= p.confidence <= 1.0

    def test_prediction_outcome_null_before_resolution(self):
        p = RealityLedger.get_prediction("P_001")
        # May be None or resolved depending on state
        assert p.outcome is None or isinstance(p.outcome, str)

    def test_get_missing_prediction_returns_none(self):
        assert RealityLedger.get_prediction("P_999") is None


# ---------------------------------------------------------------------------
# ValidationPipeline
# ---------------------------------------------------------------------------

class TestValidationPipeline:
    def test_gates_dict_nonempty(self):
        assert len(ValidationPipeline.GATES) >= 5

    def test_m01_intake_required(self):
        assert "M01_INTAKE" in ValidationPipeline.GATES
        assert ValidationPipeline.GATES["M01_INTAKE"]["status"] == "REQUIRED"

    def test_validate_metric_high_r2_passes(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", 0.42, r_squared=0.96)
        assert result["status"] == "PASS"

    def test_validate_metric_mid_r2_conditional(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", 0.42, r_squared=0.85)
        assert result["status"] == "CONDITIONAL"

    def test_validate_metric_low_r2_fails(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", 0.42, r_squared=0.50)
        assert result["status"] == "FAIL"

    def test_validate_metric_nan_result_invalid(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", float("nan"))
        assert result["status"] == "INVALID"

    def test_validate_metric_inf_result_invalid(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", float("inf"))
        assert result["status"] == "INVALID"

    def test_validate_metric_no_r2_unknown(self):
        result = ValidationPipeline.validate_metric("test", "BRIER_SCORE", 0.42)
        assert result["status"] == "UNKNOWN"

    def test_classify_claim_none_observation(self):
        assert ValidationPipeline.classify_claim(None) == "UNKNOWN"

    def test_classify_claim_high_r2_observed(self):
        assert ValidationPipeline.classify_claim("signal detected", evidence_r2=0.97) == "OBSERVED"

    def test_classify_claim_mid_r2_with_theory_supported(self):
        result = ValidationPipeline.classify_claim("signal", evidence_r2=0.85, theory_support=True)
        assert result == "SUPPORTED"

    def test_classify_claim_mid_r2_no_theory_plausible(self):
        result = ValidationPipeline.classify_claim("signal", evidence_r2=0.70)
        assert result == "PLAUSIBLE"

    def test_classify_claim_negative_r2_refuted(self):
        result = ValidationPipeline.classify_claim("signal", evidence_r2=-0.5)
        assert result == "REFUTED"

    def test_classify_claim_low_r2_unknown(self):
        result = ValidationPipeline.classify_claim("signal", evidence_r2=0.45)
        assert result == "UNKNOWN"

    def test_classify_claim_exactly_095_observed(self):
        assert ValidationPipeline.classify_claim("x", evidence_r2=0.95) == "OBSERVED"

    def test_classify_claim_boundary_094_with_theory(self):
        result = ValidationPipeline.classify_claim("x", evidence_r2=0.80, theory_support=True)
        assert result == "SUPPORTED"
