"""Tests for gpts_core.signals — time-series structure analysis."""
from __future__ import annotations

import math
import numpy as np
import pytest

from gpts_core.signals import (
    entropy_norm,
    compression_ratio,
    autocorr,
    spectral_analysis,
    motif_share,
    window_stats,
    ar2_fit,
    structure_score,
    analyze,
    step_metrics,
    generate_null,
    bootstrap_ci,
)


# ---------------------------------------------------------------------------
# entropy_norm
# ---------------------------------------------------------------------------

class TestEntropyNorm:
    def test_uniform_distribution_near_one(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 10000)
        val = entropy_norm(x, bins=32)
        assert 0.95 < val <= 1.0

    def test_constant_signal_zero(self):
        x = np.ones(200)
        val = entropy_norm(x)
        assert val == 0.0

    def test_two_values_partial_entropy(self):
        x = np.array([0.0] * 100 + [1.0] * 100)
        val = entropy_norm(x)
        assert 0.0 < val < 1.0

    def test_empty_returns_zero(self):
        x = np.array([])
        val = entropy_norm(x, bins=4)
        assert val == 0.0


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_constant_signal_highly_compressible(self):
        x = np.zeros(1000)
        ratio = compression_ratio(x)
        assert ratio < 0.5

    def test_random_signal_less_compressible(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 1000)
        ratio_rand = compression_ratio(x)
        x_const = np.zeros(1000)
        ratio_const = compression_ratio(x_const)
        assert ratio_rand > ratio_const

    def test_returns_positive_float(self):
        x = np.arange(100, dtype=float)
        r = compression_ratio(x)
        assert isinstance(r, float)
        assert r > 0.0


# ---------------------------------------------------------------------------
# autocorr
# ---------------------------------------------------------------------------

class TestAutocorr:
    def test_sine_high_autocorr(self):
        t = np.linspace(0, 4 * np.pi, 512)
        x = np.sin(t)
        lag1, acmax = autocorr(x)
        assert acmax > 0.8

    def test_constant_signal_zero(self):
        x = np.ones(100)
        lag1, acmax = autocorr(x)
        assert lag1 == 0.0
        assert acmax == 0.0

    def test_returns_two_floats(self):
        x = np.arange(50, dtype=float)
        lag1, acmax = autocorr(x, maxlag=10)
        assert isinstance(lag1, float)
        assert isinstance(acmax, float)


# ---------------------------------------------------------------------------
# spectral_analysis
# ---------------------------------------------------------------------------

class TestSpectralAnalysis:
    def test_pure_sine_detects_frequency(self):
        sr = 256.0
        freq = 10.0
        t = np.arange(512) / sr
        x = np.sin(2 * np.pi * freq * t)
        pf, conc, ent = spectral_analysis(x, sr)
        assert abs(pf - freq) < 2.0
        assert conc > 0.5

    def test_constant_returns_zeros(self):
        x = np.ones(128)
        pf, conc, ent = spectral_analysis(x, 256.0)
        assert pf == 0.0
        assert conc == 0.0

    def test_short_signal_returns_zeros(self):
        x = np.ones(4)
        pf, conc, ent = spectral_analysis(x, 256.0)
        assert pf == 0.0

    def test_returns_three_floats(self):
        x = np.random.default_rng(0).normal(0, 1, 256)
        pf, conc, ent = spectral_analysis(x, 256.0)
        assert isinstance(pf, float)
        assert isinstance(conc, float)
        assert isinstance(ent, float)


# ---------------------------------------------------------------------------
# motif_share
# ---------------------------------------------------------------------------

class TestMotifShare:
    def test_constant_sequence_near_one(self):
        x = np.ones(100)
        share = motif_share(x)
        assert share > 0.9

    def test_short_sequence_returns_zero(self):
        x = np.array([1.0, 2.0])
        share = motif_share(x, mlen=4)
        assert share == 0.0

    def test_random_sequence_positive(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        share = motif_share(x)
        assert 0.0 < share <= 1.0


# ---------------------------------------------------------------------------
# window_stats
# ---------------------------------------------------------------------------

class TestWindowStats:
    def test_too_short_returns_nans(self):
        x = np.ones(100)
        pcv, ecv, jsd = window_stats(x, 256.0, window=512)
        assert math.isnan(pcv)

    def test_long_signal_returns_finite(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 2048)
        pcv, ecv, jsd = window_stats(x, 256.0, window=256)
        assert math.isfinite(jsd)


# ---------------------------------------------------------------------------
# ar2_fit
# ---------------------------------------------------------------------------

class TestAr2Fit:
    def test_ar2_process_recovers_coefficients(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = np.zeros(n)
        a1_true, a2_true = 1.2, -0.6
        for i in range(2, n):
            x[i] = a1_true * x[i - 1] + a2_true * x[i - 2] + rng.normal(0, 0.1)
        a1, a2, r2, ef, rad = ar2_fit(x, 256.0)
        assert abs(a1 - a1_true) < 0.05
        assert abs(a2 - a2_true) < 0.05
        assert r2 > 0.5

    def test_constant_signal_returns_zeros(self):
        x = np.ones(100)
        a1, a2, r2, ef, rad = ar2_fit(x, 256.0)
        assert a1 == 0.0
        assert a2 == 0.0


# ---------------------------------------------------------------------------
# structure_score
# ---------------------------------------------------------------------------

class TestStructureScore:
    def _make_row(self, **overrides):
        base = {
            "autocorr_max": 0.0,
            "spectral_concentration": 0.0,
            "ar2_r2": 0.0,
            "motif_top_share": 0.0,
            "compression_ratio": 1.0,
            "peak_freq_cv": float("inf"),
            "segment_jsd_mean": float("inf"),
        }
        base.update(overrides)
        return base

    def test_all_low_gives_zero(self):
        row = self._make_row()
        score = structure_score(row)
        assert score == pytest.approx(0.0)

    def test_high_autocorr_raises_score(self):
        row_high = self._make_row(autocorr_max=0.95)
        row_low = self._make_row(autocorr_max=0.0)
        assert structure_score(row_high) > structure_score(row_low)

    def test_score_bounded_zero_to_one(self):
        row = self._make_row(
            autocorr_max=1.0,
            spectral_concentration=1.0,
            ar2_r2=1.0,
            motif_top_share=1.0,
            compression_ratio=0.0,
            peak_freq_cv=0.0,
            segment_jsd_mean=0.0,
        )
        score = structure_score(row)
        assert 0.0 <= score <= 1.0

    def test_nan_inf_handled_gracefully(self):
        row = self._make_row(peak_freq_cv=float("nan"), segment_jsd_mean=float("nan"))
        score = structure_score(row)
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_all_expected_keys(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 512).tolist()
        result = analyze(x)
        for key in [
            "entropy_norm", "compression_ratio", "autocorr_lag1", "autocorr_max",
            "spectral_peak_hz", "spectral_concentration", "spectral_entropy",
            "motif_top_share", "peak_freq_cv", "entropy_cv", "segment_jsd_mean",
            "ar2_a1", "ar2_a2", "ar2_r2", "estimated_freq_hz", "root_radius",
            "structure_score",
        ]:
            assert key in result

    def test_structured_signal_higher_score(self):
        sr = 256.0
        t = np.arange(2048) / sr
        structured = (np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(0).normal(0, 1, 2048)).tolist()
        noise = np.random.default_rng(1).normal(0, 1, 2048).tolist()
        score_s = analyze(structured, sr)["structure_score"]
        score_n = analyze(noise, sr)["structure_score"]
        assert score_s > score_n


# ---------------------------------------------------------------------------
# step_metrics
# ---------------------------------------------------------------------------

class TestStepMetrics:
    def test_ideal_step_response(self):
        n = 200
        t = np.linspace(0, 1, n)
        y = np.zeros(n)
        y[50:] = 1.0
        result = step_metrics(t, y)
        assert "rise_time" in result
        assert "steady_state" in result
        assert result["steady_state"] == pytest.approx(1.0, abs=0.1)

    def test_single_point_returns_empty(self):
        result = step_metrics(np.array([0.0]), np.array([0.0]))
        assert result == {}


# ---------------------------------------------------------------------------
# generate_null
# ---------------------------------------------------------------------------

class TestGenerateNull:
    @pytest.mark.parametrize("kind", ["shuffle", "gaussian", "same_marginal", "phase_scramble"])
    def test_returns_same_length(self, kind):
        x = np.arange(100, dtype=float)
        null = generate_null(x, seed=0, kind=kind)
        assert len(null) == len(x)

    def test_shuffle_preserves_values(self):
        x = np.arange(100, dtype=float)
        null = generate_null(x, seed=0, kind="shuffle")
        assert sorted(null) == sorted(x)

    def test_unknown_kind_raises(self):
        x = np.ones(10)
        with pytest.raises(ValueError, match="Unknown null kind"):
            generate_null(x, seed=0, kind="invalid")


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        result = bootstrap_ci(values, seed=42, n_boot=500)
        assert result["n"] == 100
        assert abs(result["mean"] - 3.0) < 0.1
        assert result["ci_low"] < result["mean"]
        assert result["ci_high"] > result["mean"]

    def test_empty_list_returns_nans(self):
        result = bootstrap_ci([], seed=0)
        assert result["n"] == 0
        assert math.isnan(result["mean"])
        assert math.isnan(result["ci_low"])

    def test_all_infinite_returns_nans(self):
        result = bootstrap_ci([float("inf"), float("nan")], seed=0)
        assert result["n"] == 0
