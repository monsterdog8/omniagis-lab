"""Tests for epsilon_sweep.py — ε-sweep robustness analysis."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from omniagis.epsilon_sweep import (
    SweepParams,
    run_epsilon_sweep,
)


class TestSweepParams:
    """Tests for SweepParams dataclass."""

    def test_default_params(self):
        """Test default sweep parameters."""
        params = SweepParams()

        assert params.z == 1.5
        assert params.epsilon_min == 0.02
        assert params.epsilon_max == 0.20
        assert params.n_epsilons == 10
        assert params.n_steps == 1_000_000
        assert params.x0 == 0.5
        assert params.n_min == 5
        assert params.min_tail_sample_size == 1000
        assert params.min_tail_obs == 10
        assert params.min_r_squared == 0.80
        assert params.require_plateau is False
        assert params.n_bootstrap == 200
        assert params.seed == 42

    def test_custom_params(self):
        """Test custom sweep parameters."""
        params = SweepParams(
            z=2.0,
            epsilon_min=0.01,
            epsilon_max=0.30,
            n_epsilons=5,
            n_steps=50000,
            seed=99,
        )

        assert params.z == 2.0
        assert params.epsilon_min == 0.01
        assert params.epsilon_max == 0.30
        assert params.n_epsilons == 5
        assert params.n_steps == 50000
        assert params.seed == 99


class TestRunEpsilonSweep:
    """Tests for the run_epsilon_sweep function."""

    def test_basic_sweep(self):
        """Test running a basic epsilon sweep."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=3,
            n_steps=5000,
            n_bootstrap=20,
            min_tail_sample_size=50,
        )

        result = run_epsilon_sweep(params)

        # Check top-level structure
        assert "sweep_params" in result
        assert "theory_alpha" in result
        assert "timestamp" in result
        assert "summary" in result
        assert "results" in result

        # Check theory_alpha
        expected_alpha = 1.0 / (params.z - 1.0)
        assert abs(result["theory_alpha"] - expected_alpha) < 1e-10

        # Check results array
        assert len(result["results"]) == params.n_epsilons

    def test_sweep_results_structure(self):
        """Test that each result has the correct structure."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.08,
            epsilon_max=0.12,
            n_epsilons=2,
            n_steps=3000,
            n_bootstrap=10,
            min_tail_sample_size=20,
        )

        result = run_epsilon_sweep(params)

        # Check each result row
        for res in result["results"]:
            assert "epsilon" in res
            assert "verdict" in res
            assert res["verdict"] in ["ACCEPTED", "FAIL_CLOSED"]
            assert "n_return_times" in res
            assert "checks" in res
            assert "fail_reasons" in res
            assert "power_law" in res
            assert "multi_scale_ci" in res
            assert "tail_mass" in res
            assert "plateau_detected" in res

            # Check power_law structure
            pl = res["power_law"]
            assert "alpha" in pl
            assert "r_squared" in pl
            assert "tail_sample_size" in pl
            assert "n_tail_points" in pl

            # Check multi_scale_ci structure
            ms = res["multi_scale_ci"]
            assert "scales" in ms
            assert "width" in ms
            assert "alpha_hat" in ms
            assert "lower" in ms
            assert "upper" in ms
            assert "ci_shrinks" in ms

    def test_sweep_reproducibility(self):
        """Test that sweeps with same seed are reproducible."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=3,
            n_steps=3000,
            n_bootstrap=10,
            min_tail_sample_size=20,
            seed=42,
        )

        result1 = run_epsilon_sweep(params)
        result2 = run_epsilon_sweep(params)

        # Check that verdicts are the same for all epsilon values
        for r1, r2 in zip(result1["results"], result2["results"]):
            assert r1["epsilon"] == r2["epsilon"]
            assert r1["verdict"] == r2["verdict"]
            assert r1["n_return_times"] == r2["n_return_times"]

    def test_sweep_summary_structure(self):
        """Test that summary has the correct structure."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=4,
            n_steps=5000,
            n_bootstrap=20,
            min_tail_sample_size=50,
        )

        result = run_epsilon_sweep(params)
        summary = result["summary"]

        # Check summary fields
        assert "n_epsilons" in summary
        assert "n_accepted" in summary
        assert "n_fail_closed" in summary
        assert "acceptance_rate" in summary
        assert "theory_alpha" in summary
        assert "alpha_best_scale" in summary
        assert "ci_width_best_scale" in summary
        assert "check_pass_counts" in summary

        # Check acceptance counts
        assert summary["n_accepted"] + summary["n_fail_closed"] == params.n_epsilons
        assert 0.0 <= summary["acceptance_rate"] <= 1.0

        # Check stats structure
        for stats_key in ["alpha_best_scale", "ci_width_best_scale"]:
            stats = summary[stats_key]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "n" in stats

    def test_sweep_epsilon_ordering(self):
        """Test that epsilon values are log-spaced and in order."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.01,
            epsilon_max=0.20,
            n_epsilons=5,
            n_steps=2000,
            n_bootstrap=10,
            min_tail_sample_size=10,
        )

        result = run_epsilon_sweep(params)

        # Extract epsilon values
        epsilons = [r["epsilon"] for r in result["results"]]

        # Check they are strictly increasing
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i + 1]

        # Check bounds
        assert epsilons[0] >= params.epsilon_min * 0.99  # Allow small float error
        assert epsilons[-1] <= params.epsilon_max * 1.01

    def test_sweep_shared_trajectory(self):
        """Test that all epsilon values use the same trajectory."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=3,
            n_steps=3000,
            n_bootstrap=10,
            min_tail_sample_size=20,
            seed=42,
        )

        result = run_epsilon_sweep(params)

        # With the same trajectory but different epsilon, we expect different
        # numbers of return times (larger epsilon → more returns)
        return_times = [r["n_return_times"] for r in result["results"]]

        # Generally, larger epsilon should give more returns (though not guaranteed)
        # At least verify that they are not all the same
        assert len(set(return_times)) > 1 or all(rt == 0 for rt in return_times)

    def test_sweep_output_serializable(self):
        """Test that sweep output is JSON-serializable."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=2,
            n_steps=1000,
            n_bootstrap=5,
            min_tail_sample_size=10,
        )

        result = run_epsilon_sweep(params)

        # Should not raise an exception
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Should be able to load it back
        loaded = json.loads(json_str)
        assert loaded["theory_alpha"] == result["theory_alpha"]
        assert len(loaded["results"]) == len(result["results"])

    def test_sweep_with_different_z(self):
        """Test sweeps with different z values."""
        params1 = SweepParams(z=1.5, n_epsilons=2, n_steps=2000, n_bootstrap=10)
        params2 = SweepParams(z=2.0, n_epsilons=2, n_steps=2000, n_bootstrap=10)

        result1 = run_epsilon_sweep(params1)
        result2 = run_epsilon_sweep(params2)

        # Different z should give different theory_alpha
        assert result1["theory_alpha"] != result2["theory_alpha"]
        assert abs(result1["theory_alpha"] - 2.0) < 0.1  # z=1.5 → α≈2
        assert abs(result2["theory_alpha"] - 1.0) < 0.1  # z=2.0 → α≈1

    def test_sweep_metadata_preserved(self):
        """Test that sweep parameters are preserved in output."""
        params = SweepParams(
            z=1.8,
            epsilon_min=0.03,
            epsilon_max=0.18,
            n_epsilons=4,
            n_steps=4000,
            x0=0.6,
            seed=123,
        )

        result = run_epsilon_sweep(params)

        sweep_params = result["sweep_params"]
        assert sweep_params["z"] == 1.8
        assert sweep_params["epsilon_min"] == 0.03
        assert sweep_params["epsilon_max"] == 0.18
        assert sweep_params["n_epsilons"] == 4
        assert sweep_params["n_steps"] == 4000
        assert sweep_params["x0"] == 0.6
        assert sweep_params["seed"] == 123

    def test_single_epsilon_sweep(self):
        """Test sweep with a single epsilon value."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.1,
            epsilon_max=0.1,
            n_epsilons=1,
            n_steps=2000,
            n_bootstrap=10,
            min_tail_sample_size=20,
        )

        result = run_epsilon_sweep(params)

        assert len(result["results"]) == 1
        assert abs(result["results"][0]["epsilon"] - 0.1) < 0.01

    def test_sweep_with_plateau_requirement(self):
        """Test sweep with plateau requirement."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=2,
            n_steps=3000,
            n_bootstrap=10,
            require_plateau=True,
            min_tail_sample_size=20,
        )

        result = run_epsilon_sweep(params)

        # Check that plateau detection is included
        for res in result["results"]:
            assert "plateau_detected" in res
            assert isinstance(res["plateau_detected"], bool)

    def test_sweep_increasing_returns_with_epsilon(self):
        """Test that larger epsilon generally gives more returns."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.02,
            epsilon_max=0.20,
            n_epsilons=5,
            n_steps=5000,
            n_bootstrap=10,
            min_tail_sample_size=20,
        )

        result = run_epsilon_sweep(params)

        # Extract epsilon and return times
        data = [(r["epsilon"], r["n_return_times"]) for r in result["results"]]

        # Generally, larger epsilon should give more returns
        # Check that the last epsilon has at least as many returns as the first
        # (though this is not guaranteed, it's highly likely)
        if data[0][1] > 0 and data[-1][1] > 0:
            # If both have returns, last should have more or equal
            assert data[-1][1] >= data[0][1] * 0.5  # Allow some variation


class TestSweepIntegration:
    """Integration tests for epsilon sweep."""

    def test_full_sweep_pipeline(self):
        """Test the full sweep validation pipeline."""
        params = SweepParams(
            z=1.5,
            epsilon_min=0.05,
            epsilon_max=0.15,
            n_epsilons=4,
            n_steps=8000,
            n_bootstrap=30,
            min_tail_sample_size=50,
            min_r_squared=0.7,
            seed=42,
        )

        result = run_epsilon_sweep(params)

        # Check complete structure
        assert "sweep_params" in result
        assert "theory_alpha" in result
        assert "timestamp" in result
        assert "summary" in result
        assert "results" in result

        # Check all results have all six checks
        for res in result["results"]:
            checks = res["checks"]
            assert len(checks) == 6
            assert "R2_ok" in checks
            assert "CI_width_ok" in checks
            assert "theory_in_CI" in checks
            assert "CI_shrinks" in checks
            assert "plateau" in checks
            assert "tail_mass" in checks

        # Check summary aggregates
        summary = result["summary"]
        assert summary["n_accepted"] >= 0
        assert summary["n_fail_closed"] >= 0
        assert summary["n_accepted"] + summary["n_fail_closed"] == params.n_epsilons
