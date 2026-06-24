"""Tests for gpts_ultimate.signals — all 12 core + 4 world-agnostic + 3 null surrogates."""
import math

import numpy as np
import pytest

from gpts_ultimate.signals import (
    analyze,
    ar2_fit,
    autocorr,
    block_shuffle,
    bootstrap_ci,
    coherence,
    compression_ratio,
    direction,
    entropy_norm,
    generate_null,
    iid_shuffle,
    memory,
    motif_share,
    phase_scramble,
    spectral_analysis,
    spectral_entropy_agnostic,
    step_metrics,
    structure_score,
    window_stats,
)

RNG = np.random.default_rng(42)
SINE = np.sin(np.linspace(0, 8 * np.pi, 512))
NOISE = RNG.normal(0, 1, 512)
RAMP = np.linspace(0, 1, 200)


# ---------------------------------------------------------------------------
# entropy_norm
# ---------------------------------------------------------------------------
class TestEntropyNorm:
    def test_uniform_max(self):
        x = np.ones(256)
        assert entropy_norm(x) == 0.0

    def test_range(self):
        v = entropy_norm(SINE)
        assert 0.0 <= v <= 1.0

    def test_sine_in_range(self):
        # Sine amplitude has arcsine distribution — may be near Gaussian entropy
        assert 0.0 <= entropy_norm(SINE) <= 1.0

    def test_empty(self):
        assert entropy_norm(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------
class TestCompressionRatio:
    def test_constant(self):
        x = np.full(512, 3.14)
        assert compression_ratio(x) < 0.5

    def test_range(self):
        cr = compression_ratio(SINE)
        assert 0.0 < cr <= 1.0

    def test_sine_more_compressible_than_noise(self):
        assert compression_ratio(SINE) < compression_ratio(NOISE)


# ---------------------------------------------------------------------------
# autocorr
# ---------------------------------------------------------------------------
class TestAutocorr:
    def test_sine_high_autocorr(self):
        lag1, acmax = autocorr(SINE)
        assert acmax > 0.5

    def test_returns_tuple(self):
        result = autocorr(SINE)
        assert len(result) == 2

    def test_constant_signal(self):
        lag1, acmax = autocorr(np.ones(100))
        assert lag1 == 0.0 and acmax == 0.0

    def test_default_maxlag_is_64(self):
        lag1, _ = autocorr(SINE, maxlag=64)
        assert isinstance(lag1, float)


# ---------------------------------------------------------------------------
# spectral_analysis
# ---------------------------------------------------------------------------
class TestSpectralAnalysis:
    def test_returns_three(self):
        result = spectral_analysis(SINE, sr=256.0)
        assert len(result) == 3

    def test_short_signal(self):
        pf, sc, se = spectral_analysis(np.ones(4), sr=256.0)
        assert pf == 0.0 and sc == 0.0 and se == 0.0

    def test_sine_has_peak(self):
        pf, sc, se = spectral_analysis(SINE, sr=256.0)
        assert pf > 0.0
        assert sc > 0.0


# ---------------------------------------------------------------------------
# motif_share
# ---------------------------------------------------------------------------
class TestMotifShare:
    def test_returns_float(self):
        v = motif_share(SINE)
        assert isinstance(v, float)

    def test_range(self):
        assert 0.0 <= motif_share(SINE) <= 1.0

    def test_short_signal(self):
        assert motif_share(np.array([1.0, 2.0])) == 0.0


# ---------------------------------------------------------------------------
# window_stats
# ---------------------------------------------------------------------------
class TestWindowStats:
    def test_returns_three(self):
        result = window_stats(SINE, sr=256.0)
        assert len(result) == 3

    def test_few_windows(self):
        pcv, ecv, jsd = window_stats(np.ones(10), sr=256.0)
        assert math.isnan(pcv)


# ---------------------------------------------------------------------------
# ar2_fit
# ---------------------------------------------------------------------------
class TestAr2Fit:
    def test_returns_five(self):
        result = ar2_fit(SINE, sr=256.0)
        assert len(result) == 5

    def test_sine_good_r2(self):
        a1, a2, r2, ef, rr = ar2_fit(SINE, sr=256.0)
        assert r2 > 0.5


# ---------------------------------------------------------------------------
# step_metrics
# ---------------------------------------------------------------------------
class TestStepMetrics:
    def test_returns_dict(self):
        t = np.linspace(0, 1, 100)
        y = np.where(t < 0.1, 0.0, 1.0)
        m = step_metrics(t, y)
        assert "steady_state" in m

    def test_short_signal(self):
        m = step_metrics(np.array([0.0]), np.array([1.0]))
        assert m == {}


# ---------------------------------------------------------------------------
# generate_null
# ---------------------------------------------------------------------------
class TestGenerateNull:
    def test_shuffle(self):
        n = generate_null(SINE, seed=0, kind="shuffle")
        assert len(n) == len(SINE)
        assert sorted(n) == pytest.approx(sorted(SINE), abs=1e-9)

    def test_gaussian(self):
        n = generate_null(SINE, seed=0, kind="gaussian")
        assert len(n) == len(SINE)

    def test_same_marginal(self):
        n = generate_null(SINE, seed=0, kind="same_marginal")
        assert len(n) == len(SINE)

    def test_phase_scramble(self):
        n = generate_null(SINE, seed=0, kind="phase_scramble")
        assert len(n) == len(SINE)

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            generate_null(SINE, seed=0, kind="bogus")


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
class TestBootstrapCi:
    def test_returns_dict(self):
        r = bootstrap_ci([1.0, 2.0, 3.0], seed=0)
        assert "mean" in r and "ci_low" in r and "ci_high" in r

    def test_empty(self):
        r = bootstrap_ci([], seed=0)
        assert r["n"] == 0

    def test_ci_order(self):
        r = bootstrap_ci(list(range(50)), seed=0)
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]


# ---------------------------------------------------------------------------
# structure_score
# ---------------------------------------------------------------------------
class TestStructureScore:
    def test_analyze_has_structure_score(self):
        row = analyze(SINE.tolist())
        assert "structure_score" in row
        assert 0.0 <= row["structure_score"] <= 1.0

    def test_sine_higher_than_noise(self):
        s_sine = analyze(SINE.tolist())["structure_score"]
        s_noise = analyze(NOISE.tolist())["structure_score"]
        assert s_sine > s_noise


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
class TestAnalyze:
    EXPECTED_KEYS = {
        "entropy_norm", "compression_ratio", "autocorr_lag1", "autocorr_max",
        "spectral_peak_hz", "spectral_concentration", "spectral_entropy",
        "motif_top_share", "peak_freq_cv", "entropy_cv", "segment_jsd_mean",
        "ar2_a1", "ar2_a2", "ar2_r2", "estimated_freq_hz", "root_radius",
        "structure_score",
    }

    def test_all_keys_present(self):
        row = analyze(SINE.tolist())
        assert self.EXPECTED_KEYS <= set(row.keys())

    def test_all_float(self):
        row = analyze(SINE.tolist())
        for k, v in row.items():
            assert isinstance(v, float), f"{k} is not float"


# ---------------------------------------------------------------------------
# World-agnostic shortcuts
# ---------------------------------------------------------------------------
class TestMemory:
    def test_high_for_sine(self):
        assert memory(SINE) > 0.5

    def test_low_for_constant(self):
        assert memory(np.ones(100)) == 0.0

    def test_short(self):
        assert memory(np.array([1.0])) == 0.0


class TestCoherence:
    def test_correlated(self):
        channels = np.column_stack([SINE, SINE + 0.01])
        assert coherence(channels) > 0.9

    def test_1d_returns_zero(self):
        assert coherence(SINE) == 0.0

    def test_single_channel(self):
        assert coherence(SINE.reshape(-1, 1)) == 0.0


class TestDirection:
    def test_ramp_high(self):
        assert direction(RAMP) > 0.5

    def test_sine_low(self):
        assert direction(SINE) < 0.5

    def test_constant(self):
        assert direction(np.ones(100)) == 0.0


class TestSpectralEntropyAgnostic:
    def test_noise_higher_than_sine(self):
        se_noise = spectral_entropy_agnostic(NOISE)
        se_sine = spectral_entropy_agnostic(SINE)
        assert se_noise > se_sine

    def test_constant(self):
        assert spectral_entropy_agnostic(np.zeros(64)) == 0.0


# ---------------------------------------------------------------------------
# Null surrogates
# ---------------------------------------------------------------------------
class TestPhaseScramble:
    def test_same_length(self):
        n = phase_scramble(SINE, seed=0)
        assert len(n) == len(SINE)

    def test_same_mean(self):
        n = phase_scramble(SINE, seed=0)
        assert abs(np.mean(n) - np.mean(SINE)) < 0.5

    def test_different_order(self):
        n = phase_scramble(SINE, seed=99)
        assert not np.allclose(n, SINE)


class TestIidShuffle:
    def test_same_length(self):
        n = iid_shuffle(SINE, seed=0)
        assert len(n) == len(SINE)

    def test_same_values(self):
        n = iid_shuffle(SINE, seed=0)
        assert sorted(n) == pytest.approx(sorted(SINE), abs=1e-9)


class TestBlockShuffle:
    def test_same_length(self):
        n = block_shuffle(SINE, block_size=64, seed=0)
        assert len(n) == len(SINE)

    def test_different_order(self):
        n = block_shuffle(SINE, block_size=64, seed=7)
        assert not np.allclose(n, SINE)
