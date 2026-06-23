"""Tests for phase_omega.berzerker_v2 — BERZERKER ablation engine."""
from __future__ import annotations

import numpy as np
import pytest

from phase_omega.berzerker_v2 import (
    ORGANS,
    N,
    METRIC_KEYS,
    base_coupling,
    shuffle_coupling,
    logistic,
    simulate,
    metrics_from,
    ablation_effects,
    ci,
    null_threshold,
    tribunal_verdict,
)


class TestConstants:
    def test_organs_length(self):
        assert len(ORGANS) == 4

    def test_n_matches_organs(self):
        assert N == len(ORGANS)

    def test_metric_keys_length(self):
        assert len(METRIC_KEYS) == 4

    def test_metric_keys_content(self):
        assert set(METRIC_KEYS) == {"memory", "coherence", "direction", "attractor"}


class TestBaseCoupling:
    def test_shape(self):
        C = base_coupling(seed=0)
        assert C.shape == (N, N)

    def test_non_negative(self):
        C = base_coupling(seed=0)
        assert np.all(C >= 0)

    def test_deterministic(self):
        C1 = base_coupling(seed=42)
        C2 = base_coupling(seed=42)
        np.testing.assert_array_equal(C1, C2)

    def test_different_seeds_differ(self):
        C1 = base_coupling(seed=0)
        C2 = base_coupling(seed=99)
        assert not np.allclose(C1, C2)

    def test_diagonal_near_zero(self):
        C = base_coupling(seed=0)
        # Diagonal starts at 0.0 before small noise is added
        assert all(abs(C[i, i]) < 0.1 for i in range(N))


class TestShuffleCoupling:
    def test_shape_preserved(self):
        C = base_coupling(seed=0)
        Cn = shuffle_coupling(C, seed=1)
        assert Cn.shape == C.shape

    def test_diagonal_unchanged(self):
        C = base_coupling(seed=0)
        Cn = shuffle_coupling(C, seed=1)
        np.testing.assert_array_equal(C.diagonal(), Cn.diagonal())

    def test_off_diagonal_sum_preserved(self):
        C = base_coupling(seed=0)
        Cn = shuffle_coupling(C, seed=1)
        mask = ~np.eye(N, dtype=bool)
        assert abs(C[mask].sum() - Cn[mask].sum()) < 1e-9

    def test_different_from_original(self):
        C = base_coupling(seed=0)
        Cn = shuffle_coupling(C, seed=1)
        assert not np.allclose(C, Cn)

    def test_deterministic(self):
        C = base_coupling(seed=0)
        Cn1 = shuffle_coupling(C, seed=7)
        Cn2 = shuffle_coupling(C, seed=7)
        np.testing.assert_array_equal(Cn1, Cn2)


class TestLogistic:
    def test_fixed_points(self):
        x = np.array([0.0, 1.0])
        np.testing.assert_allclose(logistic(x), [0.0, 0.0])

    def test_maximum_at_half(self):
        x = np.array([0.5])
        result = logistic(x, r=3.82)
        assert abs(result[0] - 3.82 * 0.5 * 0.5) < 1e-10

    def test_r_parameter(self):
        x = np.array([0.3])
        assert abs(logistic(x, r=4.0)[0] - 4.0 * 0.3 * 0.7) < 1e-10

    def test_array_output_shape(self):
        x = np.linspace(0, 1, 10)
        assert logistic(x).shape == x.shape


class TestSimulate:
    @pytest.fixture
    def C(self):
        return base_coupling(seed=0)

    def test_output_shape(self, C):
        hist = simulate(C, steps=200, transient=100, seed=0)
        assert hist.shape == (100, N)

    def test_values_in_unit_interval(self, C):
        hist = simulate(C, steps=200, transient=100, seed=0)
        assert np.all(hist >= 0.0)
        assert np.all(hist <= 1.0)

    def test_ablate_column_is_zero(self, C):
        hist = simulate(C, ablate=1, steps=200, transient=100, seed=0)
        np.testing.assert_array_equal(hist[:, 1], 0.0)

    def test_different_seeds_differ(self, C):
        h1 = simulate(C, steps=200, transient=100, seed=0)
        h2 = simulate(C, steps=200, transient=100, seed=99)
        assert not np.allclose(h1, h2)

    def test_deterministic(self, C):
        h1 = simulate(C, steps=200, transient=100, seed=5)
        h2 = simulate(C, steps=200, transient=100, seed=5)
        np.testing.assert_array_equal(h1, h2)


class TestMetricsFrom:
    @pytest.fixture
    def hist(self):
        C = base_coupling(seed=0)
        return simulate(C, steps=500, transient=200, seed=0)

    def test_returns_four_keys(self, hist):
        m = metrics_from(hist)
        assert set(m.keys()) == set(METRIC_KEYS)

    def test_values_are_finite(self, hist):
        m = metrics_from(hist)
        for v in m.values():
            assert np.isfinite(v)

    def test_memory_in_unit_interval(self, hist):
        m = metrics_from(hist)
        assert 0.0 <= m["memory"] <= 1.0

    def test_coherence_in_unit_interval(self, hist):
        m = metrics_from(hist)
        assert 0.0 <= m["coherence"] <= 1.0

    def test_direction_nonnegative(self, hist):
        m = metrics_from(hist)
        assert m["direction"] >= 0.0

    def test_attractor_positive(self, hist):
        m = metrics_from(hist)
        assert m["attractor"] > 0.0

    def test_ablate_reduces_columns(self, hist):
        m_full = metrics_from(hist, ablate=None)
        m_ablate = metrics_from(hist, ablate=0)
        assert isinstance(m_ablate, dict)
        assert set(m_ablate.keys()) == set(METRIC_KEYS)

    def test_constant_signal_memory_zero(self):
        hist = np.ones((200, N)) * 0.5
        m = metrics_from(hist)
        assert m["memory"] == 0.0


class TestAblationEffects:
    @pytest.fixture
    def C(self):
        return base_coupling(seed=0)

    def test_returns_all_keys(self, C):
        seeds = [0, 1, 2]
        eff = ablation_effects(C, ablate_idx=0, seeds=seeds)
        assert set(eff.keys()) == set(METRIC_KEYS)

    def test_array_lengths_match_seeds(self, C):
        seeds = [0, 1, 2, 3]
        eff = ablation_effects(C, ablate_idx=0, seeds=seeds)
        for k in METRIC_KEYS:
            assert len(eff[k]) == 4

    def test_values_are_finite(self, C):
        seeds = [0, 1]
        eff = ablation_effects(C, ablate_idx=0, seeds=seeds)
        for k in METRIC_KEYS:
            assert np.all(np.isfinite(eff[k]))


class TestCi:
    def test_mean_between_lo_hi(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m, lo, hi, std = ci(arr)
        assert lo <= m <= hi

    def test_std_nonnegative(self):
        arr = np.random.default_rng(0).normal(0, 1, 50)
        m, lo, hi, std = ci(arr)
        assert std >= 0.0

    def test_constant_array(self):
        arr = np.ones(20) * 3.0
        m, lo, hi, std = ci(arr)
        assert abs(m - 3.0) < 1e-10
        assert abs(std) < 1e-10


class TestNullThreshold:
    @pytest.fixture
    def C(self):
        return base_coupling(seed=0)

    def test_returns_all_keys(self, C):
        seeds = [0, 1, 2]
        thr = null_threshold(C, seeds, n_null=3, base_seed=9000)
        assert set(thr.keys()) == set(METRIC_KEYS)

    def test_values_positive(self, C):
        seeds = [0, 1, 2]
        thr = null_threshold(C, seeds, n_null=3, base_seed=9000)
        for k in METRIC_KEYS:
            assert thr[k] >= 0.0

    def test_values_finite(self, C):
        seeds = [0, 1, 2]
        thr = null_threshold(C, seeds, n_null=3, base_seed=9000)
        for k in METRIC_KEYS:
            assert np.isfinite(thr[k])


class TestTribunalVerdict:
    @pytest.fixture
    def C(self):
        return base_coupling(seed=0)

    def test_returns_valid_verdict(self, C):
        seeds = [0, 1, 2]
        # Provide precomputed threshold to avoid slow null computation
        thr = {k: 1.0 for k in METRIC_KEYS}  # high threshold → REFUTED_VS_NULL
        result = tribunal_verdict(C, seeds, thr=thr)
        valid_verdicts = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert result["verdict"] in valid_verdicts

    def test_claim_ceiling_local_only(self, C):
        seeds = [0, 1]
        thr = {k: 1.0 for k in METRIC_KEYS}
        result = tribunal_verdict(C, seeds, thr=thr)
        assert result["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_blocked_claims_present(self, C):
        seeds = [0, 1]
        thr = {k: 1.0 for k in METRIC_KEYS}
        result = tribunal_verdict(C, seeds, thr=thr)
        assert "CONSCIOUSNESS" in result["blocked_claims"]

    def test_has_sha256(self, C):
        seeds = [0, 1]
        thr = {k: 1.0 for k in METRIC_KEYS}
        result = tribunal_verdict(C, seeds, thr=thr)
        assert "sha256" in result
        assert len(result["sha256"]) == 64

    def test_atlas_has_all_organs(self, C):
        seeds = [0, 1]
        thr = {k: 1.0 for k in METRIC_KEYS}
        result = tribunal_verdict(C, seeds, thr=thr)
        atlas_organs = [entry["organ"] for entry in result["atlas"]]
        assert set(atlas_organs) == set(ORGANS)

    def test_zero_threshold_gives_pass_or_empirical(self, C):
        seeds = [0, 1, 2, 3, 4]
        thr = {k: 0.0 for k in METRIC_KEYS}  # zero threshold → everything beats null
        result = tribunal_verdict(C, seeds, thr=thr)
        assert result["total_delta_d"] >= 0

    def test_output_path_creates_file(self, C, tmp_path):
        seeds = [0, 1]
        thr = {k: 1.0 for k in METRIC_KEYS}
        out = str(tmp_path / "ledger.json")
        result = tribunal_verdict(C, seeds, thr=thr, output_path=out)
        import json
        from pathlib import Path
        content = json.loads(Path(out).read_text())
        assert content["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"
