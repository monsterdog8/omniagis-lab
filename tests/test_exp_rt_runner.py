"""Tests for exp_rt_runner.py — experiment runner for return-time validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omniagis.exp_rt_runner import (
    ExperimentParams,
    generate_trajectory,
    pomeau_manneville_step,
    run_experiment,
)


class TestPomeauMannevilleMap:
    """Tests for the Pomeau-Manneville map."""

    def test_single_step(self):
        """Test a single step of the PM map."""
        x = 0.5
        z = 1.5
        result = pomeau_manneville_step(x, z)

        # f(0.5) = (0.5 + 0.5^1.5) mod 1 ≈ 0.8536
        expected = (0.5 + 0.5**1.5) % 1.0
        assert abs(result - expected) < 1e-10

    def test_modulo_operation(self):
        """Test that result is always in [0, 1)."""
        x = 0.9
        z = 2.0
        result = pomeau_manneville_step(x, z)

        assert 0.0 <= result < 1.0

    def test_fixed_point_at_zero(self):
        """Test that x=0 is a fixed point."""
        x = 0.0
        z = 1.5
        result = pomeau_manneville_step(x, z)

        assert result == 0.0


class TestGenerateTrajectory:
    """Tests for trajectory generation."""

    def test_basic_trajectory(self):
        """Test generating a basic trajectory."""
        z = 1.5
        n_steps = 100
        x0 = 0.5

        traj = generate_trajectory(z, n_steps, x0)

        assert traj.shape == (n_steps,)
        assert traj[0] == x0
        assert np.all((traj >= 0.0) & (traj < 1.0))

    def test_deterministic(self):
        """Test that trajectory generation is deterministic."""
        z = 1.5
        n_steps = 100
        x0 = 0.5

        traj1 = generate_trajectory(z, n_steps, x0)
        traj2 = generate_trajectory(z, n_steps, x0)

        np.testing.assert_array_equal(traj1, traj2)

    def test_different_z_values(self):
        """Test trajectories with different z values."""
        n_steps = 50
        x0 = 0.5

        traj1 = generate_trajectory(1.5, n_steps, x0)
        traj2 = generate_trajectory(2.0, n_steps, x0)

        # Different z should give different trajectories
        assert not np.allclose(traj1, traj2)

    def test_different_initial_conditions(self):
        """Test trajectories with different initial conditions."""
        z = 1.5
        n_steps = 50

        traj1 = generate_trajectory(z, n_steps, x0=0.3)
        traj2 = generate_trajectory(z, n_steps, x0=0.7)

        # Different x0 should give different trajectories
        assert not np.allclose(traj1, traj2)

    def test_single_step_trajectory(self):
        """Test trajectory with a single step."""
        z = 1.5
        x0 = 0.5

        traj = generate_trajectory(z, 1, x0)

        assert traj.shape == (1,)
        assert traj[0] == x0


class TestExperimentParams:
    """Tests for ExperimentParams dataclass."""

    def test_default_params(self):
        """Test default experiment parameters."""
        params = ExperimentParams()

        assert params.z == 1.5
        assert params.epsilon == 0.1
        assert params.n_steps == 1_000_000
        assert params.x0 == 0.5
        assert params.n_min == 5
        assert params.min_tail_sample_size == 1000
        assert params.min_tail_obs == 10
        assert params.min_r_squared == 0.80
        assert params.require_plateau is False
        assert params.n_bootstrap == 500
        assert params.seed == 42

    def test_custom_params(self):
        """Test custom experiment parameters."""
        params = ExperimentParams(
            z=2.0,
            epsilon=0.05,
            n_steps=10000,
            x0=0.3,
            seed=123,
        )

        assert params.z == 2.0
        assert params.epsilon == 0.05
        assert params.n_steps == 10000
        assert params.x0 == 0.3
        assert params.seed == 123


class TestRunExperiment:
    """Tests for the run_experiment function."""

    def test_basic_experiment(self):
        """Test running a basic experiment."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.1,
            n_steps=10000,
            n_bootstrap=50,
            min_tail_sample_size=100,
        )

        result = run_experiment(params)

        # Check top-level structure
        assert "experiment" in result
        assert "theory_alpha" in result
        assert "timestamp" in result
        assert "report" in result

        # Check theory_alpha calculation
        expected_alpha = 1.0 / (params.z - 1.0)
        assert abs(result["theory_alpha"] - expected_alpha) < 1e-10

        # Check report structure
        report = result["report"]
        assert "verdict" in report
        assert report["verdict"] in ["ACCEPTED", "FAIL_CLOSED"]
        assert "n_return_times" in report
        assert "checks" in report
        assert "fail_reasons" in report

    def test_experiment_reproducibility(self):
        """Test that experiments with same seed are reproducible."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.1,
            n_steps=5000,
            n_bootstrap=20,
            min_tail_sample_size=50,
            seed=42,
        )

        result1 = run_experiment(params)
        result2 = run_experiment(params)

        # Verdicts should be the same
        assert result1["report"]["verdict"] == result2["report"]["verdict"]

        # Number of return times should be the same (deterministic trajectory)
        assert result1["report"]["n_return_times"] == result2["report"]["n_return_times"]

    def test_experiment_with_different_z(self):
        """Test experiments with different z values."""
        params1 = ExperimentParams(z=1.5, n_steps=5000, n_bootstrap=20)
        params2 = ExperimentParams(z=2.0, n_steps=5000, n_bootstrap=20)

        result1 = run_experiment(params1)
        result2 = run_experiment(params2)

        # Different z should give different theory_alpha
        assert result1["theory_alpha"] != result2["theory_alpha"]
        assert abs(result1["theory_alpha"] - 2.0) < 0.1  # z=1.5 → α≈2
        assert abs(result2["theory_alpha"] - 1.0) < 0.1  # z=2.0 → α≈1

    def test_experiment_output_serializable(self):
        """Test that experiment output is JSON-serializable."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.1,
            n_steps=1000,
            n_bootstrap=10,
            min_tail_sample_size=10,
        )

        result = run_experiment(params)

        # Should not raise an exception
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Should be able to load it back
        loaded = json.loads(json_str)
        assert loaded["theory_alpha"] == result["theory_alpha"]

    def test_experiment_with_plateau_requirement(self):
        """Test experiment with plateau requirement."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.1,
            n_steps=5000,
            n_bootstrap=20,
            require_plateau=True,
            min_tail_sample_size=50,
        )

        result = run_experiment(params)

        # Check that plateau check is in the report
        assert "plateau" in result["report"]["checks"]

    def test_experiment_metadata(self):
        """Test that experiment metadata is captured correctly."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.05,
            n_steps=8000,
            x0=0.7,
            seed=99,
        )

        result = run_experiment(params)

        # Check experiment parameters are preserved
        exp_params = result["experiment"]
        assert exp_params["z"] == 1.5
        assert exp_params["epsilon"] == 0.05
        assert exp_params["n_steps"] == 8000
        assert exp_params["x0"] == 0.7
        assert exp_params["seed"] == 99

    def test_small_epsilon(self):
        """Test experiment with small epsilon (fewer returns)."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.01,  # Very small target set
            n_steps=5000,
            n_bootstrap=20,
            min_tail_sample_size=10,
        )

        result = run_experiment(params)

        # Should still produce a valid result
        assert "verdict" in result["report"]
        assert "n_return_times" in result["report"]

    def test_large_epsilon(self):
        """Test experiment with large epsilon (more returns)."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.3,  # Larger target set
            n_steps=5000,
            n_bootstrap=20,
            min_tail_sample_size=50,
        )

        result = run_experiment(params)

        # Should produce more return times
        assert result["report"]["n_return_times"] > 0


class TestExperimentIntegration:
    """Integration tests for the experiment runner."""

    def test_full_pipeline_small_scale(self):
        """Test the full validation pipeline at small scale."""
        params = ExperimentParams(
            z=1.5,
            epsilon=0.1,
            n_steps=10000,
            n_bootstrap=50,
            min_tail_sample_size=100,
            min_r_squared=0.7,
            seed=42,
        )

        result = run_experiment(params)

        # Check all six checks are present
        checks = result["report"]["checks"]
        assert len(checks) == 6
        assert "R2_ok" in checks
        assert "CI_width_ok" in checks
        assert "theory_in_CI" in checks
        assert "CI_shrinks" in checks
        assert "plateau" in checks
        assert "tail_mass" in checks

        # Check diagnostics are present
        assert "diagnostics" in result["report"]
        diag = result["report"]["diagnostics"]
        assert "power_law" in diag
        assert "multi_scale_ci" in diag
        assert "plateau" in diag
