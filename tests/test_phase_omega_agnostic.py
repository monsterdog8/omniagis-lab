"""Tests for phase_omega.agnostic — world-agnostic metrics and null surrogates."""
from __future__ import annotations

import numpy as np
import pytest

from phase_omega.agnostic import (
    WorldAgnosticMetrics,
    WorldAgnosticNulls,
    compute_metrics,
    tribunal_verdict,
)


# ---------------------------------------------------------------------------
# WorldAgnosticMetrics.memory
# ---------------------------------------------------------------------------

class TestMemory:
    def test_constant_signal_zero(self):
        s = np.ones(200)
        assert WorldAgnosticMetrics.memory(s) == 0.0

    def test_autocorrelated_signal_positive(self):
        t = np.linspace(0, 20 * np.pi, 500)
        s = np.sin(t)
        m = WorldAgnosticMetrics.memory(s)
        assert m > 0.5

    def test_iid_noise_low(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 2000)
        m = WorldAgnosticMetrics.memory(s)
        assert m < 0.2

    def test_returns_float(self):
        s = np.arange(100, dtype=float)
        m = WorldAgnosticMetrics.memory(s)
        assert isinstance(m, float)

    def test_too_short_returns_zero(self):
        s = np.array([1.0])
        assert WorldAgnosticMetrics.memory(s) == 0.0

    def test_in_unit_interval(self):
        rng = np.random.default_rng(42)
        s = rng.normal(0, 1, 300)
        m = WorldAgnosticMetrics.memory(s)
        assert 0.0 <= m <= 1.0


# ---------------------------------------------------------------------------
# WorldAgnosticMetrics.coherence
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_correlated_channels_high(self):
        s = np.sin(np.linspace(0, 10 * np.pi, 300))
        channels = np.column_stack([s, s + 0.01 * np.random.default_rng(0).normal(0, 1, 300)])
        c = WorldAgnosticMetrics.coherence(channels)
        assert c > 0.9

    def test_independent_channels_lower(self):
        rng = np.random.default_rng(0)
        channels = rng.normal(0, 1, (300, 4))
        c = WorldAgnosticMetrics.coherence(channels)
        assert c < 0.5

    def test_single_column_zero(self):
        s = np.ones((100, 1))
        c = WorldAgnosticMetrics.coherence(s)
        assert c == 0.0

    def test_1d_array_zero(self):
        s = np.arange(50, dtype=float)
        c = WorldAgnosticMetrics.coherence(s)
        assert c == 0.0

    def test_returns_float_in_unit_interval(self):
        rng = np.random.default_rng(5)
        channels = rng.normal(0, 1, (200, 3))
        c = WorldAgnosticMetrics.coherence(channels)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# WorldAgnosticMetrics.direction
# ---------------------------------------------------------------------------

class TestDirection:
    def test_monotone_increasing_positive(self):
        s = np.linspace(0, 1, 100)
        d = WorldAgnosticMetrics.direction(s)
        assert d > 0.9

    def test_oscillating_near_zero(self):
        t = np.linspace(0, 10 * np.pi, 500)
        s = np.sin(t)
        d = WorldAgnosticMetrics.direction(s)
        assert d < 0.1

    def test_constant_zero(self):
        s = np.ones(100)
        d = WorldAgnosticMetrics.direction(s)
        assert d == 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 200)
        assert WorldAgnosticMetrics.direction(s) >= 0.0


# ---------------------------------------------------------------------------
# WorldAgnosticMetrics.spectral_entropy
# ---------------------------------------------------------------------------

class TestSpectralEntropy:
    def test_pure_sine_low_entropy(self):
        t = np.linspace(0, 20 * np.pi, 512)
        s = np.sin(t)
        se = WorldAgnosticMetrics.spectral_entropy(s)
        noise_se = WorldAgnosticMetrics.spectral_entropy(
            np.random.default_rng(0).normal(0, 1, 512)
        )
        assert se < noise_se

    def test_noise_higher_entropy(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, 512)
        se = WorldAgnosticMetrics.spectral_entropy(noise)
        assert se > 0.0

    def test_constant_signal_zero(self):
        s = np.ones(128)
        se = WorldAgnosticMetrics.spectral_entropy(s)
        assert se == 0.0

    def test_returns_float(self):
        s = np.random.default_rng(0).normal(0, 1, 256)
        se = WorldAgnosticMetrics.spectral_entropy(s)
        assert isinstance(se, float)


# ---------------------------------------------------------------------------
# WorldAgnosticNulls.phase_scramble
# ---------------------------------------------------------------------------

class TestPhaseScramble:
    def test_same_length(self):
        s = np.random.default_rng(0).normal(0, 1, 200)
        null = WorldAgnosticNulls.phase_scramble(s, seed=42)
        assert len(null) == len(s)

    def test_different_from_original(self):
        s = np.sin(np.linspace(0, 4 * np.pi, 128))
        null = WorldAgnosticNulls.phase_scramble(s, seed=1)
        assert not np.allclose(s, null)

    def test_deterministic_with_seed(self):
        s = np.random.default_rng(0).normal(0, 1, 100)
        n1 = WorldAgnosticNulls.phase_scramble(s, seed=7)
        n2 = WorldAgnosticNulls.phase_scramble(s, seed=7)
        np.testing.assert_array_equal(n1, n2)

    def test_preserves_power_spectrum_envelope(self):
        t = np.linspace(0, 10 * np.pi, 512)
        s = np.sin(t) + 0.1 * np.random.default_rng(0).normal(0, 1, 512)
        null = WorldAgnosticNulls.phase_scramble(s, seed=0)
        # Magnitudes should be preserved
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(s)),
            np.abs(np.fft.rfft(null)),
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# WorldAgnosticNulls.iid_shuffle
# ---------------------------------------------------------------------------

class TestIidShuffle:
    def test_same_length(self):
        s = np.arange(100, dtype=float)
        null = WorldAgnosticNulls.iid_shuffle(s, seed=0)
        assert len(null) == len(s)

    def test_same_values_sorted(self):
        s = np.arange(50, dtype=float)
        null = WorldAgnosticNulls.iid_shuffle(s, seed=42)
        np.testing.assert_array_equal(sorted(null), sorted(s))

    def test_different_from_original(self):
        s = np.arange(100, dtype=float)
        null = WorldAgnosticNulls.iid_shuffle(s, seed=0)
        assert not np.array_equal(null, s)

    def test_deterministic_with_seed(self):
        s = np.arange(50, dtype=float)
        n1 = WorldAgnosticNulls.iid_shuffle(s, seed=5)
        n2 = WorldAgnosticNulls.iid_shuffle(s, seed=5)
        np.testing.assert_array_equal(n1, n2)


# ---------------------------------------------------------------------------
# WorldAgnosticNulls.block_shuffle
# ---------------------------------------------------------------------------

class TestBlockShuffle:
    def test_same_length(self):
        s = np.arange(300, dtype=float)
        null = WorldAgnosticNulls.block_shuffle(s, block_size=50, seed=0)
        assert len(null) == len(s)

    def test_contains_same_values(self):
        s = np.sin(np.linspace(0, 4 * np.pi, 200))
        null = WorldAgnosticNulls.block_shuffle(s, block_size=20, seed=0)
        np.testing.assert_allclose(sorted(null), sorted(s), atol=1e-10)

    def test_deterministic_with_seed(self):
        s = np.arange(100, dtype=float)
        n1 = WorldAgnosticNulls.block_shuffle(s, block_size=10, seed=3)
        n2 = WorldAgnosticNulls.block_shuffle(s, block_size=10, seed=3)
        np.testing.assert_array_equal(n1, n2)

    def test_block_size_equal_length_unchanged(self):
        s = np.arange(50, dtype=float)
        null = WorldAgnosticNulls.block_shuffle(s, block_size=50, seed=0)
        # Only one block → no shuffling possible, remainder is empty
        assert len(null) == len(s)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_four_keys(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 200)
        m = compute_metrics(s)
        assert set(m.keys()) == {"memory", "coherence", "direction", "spectral_entropy"}

    def test_handles_nan_inf(self):
        s = np.array([float("nan"), 1.0, float("inf"), 0.5] * 50)
        m = compute_metrics(s)
        for v in m.values():
            assert np.isfinite(v)

    def test_with_channels(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 200)
        channels = rng.normal(0, 1, (200, 4))
        m = compute_metrics(s, all_channels=channels)
        assert "coherence" in m


# ---------------------------------------------------------------------------
# tribunal_verdict
# ---------------------------------------------------------------------------

class TestTribunalVerdict:
    def _null_dist(self, p95=0.5):
        return {
            "memory": {"p95": p95, "p05": 0.0, "mean": 0.3},
            "coherence": {"p95": p95, "p05": 0.0, "mean": 0.3},
            "direction": {"p95": p95, "p05": 0.0, "mean": 0.1},
            "spectral_entropy": {"p95": p95, "p05": 0.0, "mean": 2.0},
        }

    def test_valid_verdicts(self):
        metrics = {"memory": 0.1, "coherence": 0.1, "direction": 0.0, "spectral_entropy": 1.0}
        null_dist = self._null_dist(p95=0.5)
        result = tribunal_verdict(metrics, null_dist)
        valid = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert result["verdict"] in valid

    def test_all_below_null_refuted(self):
        metrics = {"memory": 0.0, "coherence": 0.0, "direction": 0.0, "spectral_entropy": 0.0}
        null_dist = self._null_dist(p95=0.9)
        result = tribunal_verdict(metrics, null_dist)
        assert result["verdict"] == "REFUTED_VS_NULL"
        assert result["total_delta_d"] == 0

    def test_all_above_null_pass_local(self):
        metrics = {"memory": 1.0, "coherence": 1.0, "direction": 1.0, "spectral_entropy": 10.0}
        null_dist = self._null_dist(p95=0.1)
        result = tribunal_verdict(metrics, null_dist)
        assert result["verdict"] == "PASS_LOCAL"
        assert result["total_delta_d"] == 4

    def test_production_always_unlocked_false(self):
        metrics = {"memory": 1.0, "coherence": 1.0, "direction": 1.0, "spectral_entropy": 10.0}
        null_dist = self._null_dist(p95=0.1)
        result = tribunal_verdict(metrics, null_dist)
        assert result["production_unlocked"] is False

    def test_claim_ceiling_local_only(self):
        metrics = {"memory": 0.0}
        null_dist = {"memory": {"p95": 0.9}}
        result = tribunal_verdict(metrics, null_dist)
        assert result["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_empty_metrics_refuted(self):
        result = tribunal_verdict({}, {})
        assert result["verdict"] == "REFUTED_VS_NULL"
        assert result["total_delta_d"] == 0
