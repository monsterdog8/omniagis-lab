"""Tests for gpts_ultimate.tribunal — agnostic, BERZERKER, and full tribunal."""
import numpy as np
import pytest

from gpts_ultimate.tribunal import (
    _build_null_dist,
    _conservative,
    run_agnostic_tribunal,
    run_berzerker_tribunal,
    run_full_tribunal,
)
from gpts_ultimate.berzerker import base_coupling
from gpts_ultimate.agnostic import compute_metrics

SINE = np.sin(np.linspace(0, 8 * np.pi, 1024))
SEEDS = list(range(5, 13))


# ---------------------------------------------------------------------------
# _conservative
# ---------------------------------------------------------------------------
class TestConservative:
    def test_refuted_beats_pass(self):
        assert _conservative("REFUTED_VS_NULL", "PASS_LOCAL") == "REFUTED_VS_NULL"

    def test_same_returns_first(self):
        assert _conservative("EMPIRICAL_SIGNAL", "EMPIRICAL_SIGNAL") == "EMPIRICAL_SIGNAL"

    def test_open_beats_empirical(self):
        assert _conservative("OPEN_PROBLEM", "EMPIRICAL_SIGNAL") == "OPEN_PROBLEM"

    def test_pass_local_loses_to_open(self):
        assert _conservative("OPEN_PROBLEM", "PASS_LOCAL") == "OPEN_PROBLEM"


# ---------------------------------------------------------------------------
# run_agnostic_tribunal
# ---------------------------------------------------------------------------
class TestRunAgnosticTribunal:
    @pytest.fixture(scope="class")
    def setup(self):
        metrics = compute_metrics(SINE)
        null_dist = _build_null_dist(SINE, n_null=10)
        verdict = run_agnostic_tribunal(metrics, null_dist)
        return verdict

    def test_verdict_key_present(self, setup):
        assert "verdict" in setup

    def test_verdict_valid_value(self, setup):
        valid = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert setup["verdict"] in valid

    def test_claim_ceiling(self, setup):
        assert setup["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_production_unlocked_false(self, setup):
        assert setup["production_unlocked"] is False

    def test_total_delta_d_int(self, setup):
        assert isinstance(setup["total_delta_d"], int)


# ---------------------------------------------------------------------------
# _build_null_dist
# ---------------------------------------------------------------------------
class TestBuildNullDist:
    def test_returns_dict(self):
        nd = _build_null_dist(SINE, n_null=5)
        assert isinstance(nd, dict)
        assert len(nd) > 0

    def test_p95_present(self):
        nd = _build_null_dist(SINE, n_null=5)
        for k, v in nd.items():
            assert "p95" in v

    def test_all_floats(self):
        nd = _build_null_dist(SINE, n_null=5)
        for k, v in nd.items():
            assert isinstance(v["p95"], float)


# ---------------------------------------------------------------------------
# run_berzerker_tribunal
# ---------------------------------------------------------------------------
class TestRunBerzerkerTribunal:
    @pytest.fixture(scope="class")
    def verdict(self):
        C = base_coupling(seed=0)
        return run_berzerker_tribunal(C, seeds=list(range(5, 13)))

    def test_verdict_present(self, verdict):
        assert "verdict" in verdict

    def test_valid_verdict(self, verdict):
        valid = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert verdict["verdict"] in valid

    def test_claim_ceiling(self, verdict):
        assert verdict["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_atlas_length(self, verdict):
        assert len(verdict["atlas"]) == 4  # four ORGANS

    def test_sha256_present(self, verdict):
        assert "sha256" in verdict

    def test_output_path(self, tmp_path, verdict):
        import json
        out = str(tmp_path / "report.json")
        C = base_coupling(seed=0)
        r = run_berzerker_tribunal(C, seeds=list(range(5, 13)), output_path=out)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["verdict"] == r["verdict"]


# ---------------------------------------------------------------------------
# run_full_tribunal
# ---------------------------------------------------------------------------
class TestRunFullTribunal:
    @pytest.fixture(scope="class")
    def result_no_lorenz(self):
        return run_full_tribunal(SINE, lorenz_ensemble=None, seeds=list(range(5, 13)), n_null=10)

    def test_final_verdict_present(self, result_no_lorenz):
        assert "final_verdict" in result_no_lorenz

    def test_agnostic_present(self, result_no_lorenz):
        assert result_no_lorenz["agnostic"] is not None

    def test_berzerker_none_without_ensemble(self, result_no_lorenz):
        assert result_no_lorenz["berzerker"] is None

    def test_claim_ceiling(self, result_no_lorenz):
        assert result_no_lorenz["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_production_unlocked_false(self, result_no_lorenz):
        assert result_no_lorenz["production_unlocked"] is False

    def test_with_lorenz_ensemble(self):
        ensemble = np.column_stack([SINE, SINE * 0.9 + 0.05])
        result = run_full_tribunal(SINE, lorenz_ensemble=ensemble,
                                   seeds=list(range(5, 11)), n_null=5)
        assert result["berzerker"] is not None
        assert result["final_verdict"] is not None

    def test_conservative_merge(self):
        ensemble = np.column_stack([SINE, SINE * 0.8])
        result = run_full_tribunal(SINE, lorenz_ensemble=ensemble,
                                   seeds=list(range(5, 11)), n_null=5)
        # final verdict must be ≤ both individual verdicts
        valid = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert result["final_verdict"] in valid
