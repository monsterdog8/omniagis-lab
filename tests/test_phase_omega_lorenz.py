"""Tests for phase_omega.lorenz — FastLorenzGenerator RK4."""
from __future__ import annotations

import numpy as np
import pytest

from phase_omega.lorenz import FastLorenzGenerator


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params(self):
        gen = FastLorenzGenerator()
        assert gen.sigma == 10.0
        assert gen.rho == 28.0
        assert abs(gen.beta - 8 / 3) < 1e-10

    def test_custom_params(self):
        gen = FastLorenzGenerator(sigma=5.0, rho=20.0, beta=1.0, dt=0.01)
        assert gen.sigma == 5.0
        assert gen.rho == 20.0
        assert gen.beta == 1.0
        assert gen.dt == 0.01

    def test_dt_stored(self):
        gen = FastLorenzGenerator(dt=0.02)
        assert gen.dt == 0.02

    def test_t_max_stored(self):
        gen = FastLorenzGenerator(t_max=100.0)
        assert gen.t_max == 100.0


# ---------------------------------------------------------------------------
# rk4_step
# ---------------------------------------------------------------------------

class TestRk4Step:
    def test_returns_3d_array(self):
        gen = FastLorenzGenerator()
        state = np.array([1.0, 1.0, 1.0])
        new_state = gen.rk4_step(state, 0.01)
        assert new_state.shape == (3,)

    def test_state_changes(self):
        gen = FastLorenzGenerator()
        state = np.array([1.0, 2.0, 3.0])
        new_state = gen.rk4_step(state, 0.01)
        assert not np.allclose(state, new_state)

    def test_zero_dt_returns_same(self):
        gen = FastLorenzGenerator()
        state = np.array([1.0, 2.0, 3.0])
        new_state = gen.rk4_step(state, 0.0)
        np.testing.assert_allclose(new_state, state, atol=1e-14)

    def test_output_finite(self):
        gen = FastLorenzGenerator()
        state = np.array([0.0, 1.0, 25.0])
        new_state = gen.rk4_step(state, 0.05)
        assert np.all(np.isfinite(new_state))

    def test_origin_fixed_point_near_zero(self):
        gen = FastLorenzGenerator()
        state = np.array([0.0, 0.0, 0.0])
        new_state = gen.rk4_step(state, 0.01)
        np.testing.assert_allclose(new_state, [0.0, 0.0, 0.0], atol=1e-14)


# ---------------------------------------------------------------------------
# generate_single_trajectory
# ---------------------------------------------------------------------------

class TestGenerateSingleTrajectory:
    @pytest.fixture
    def gen(self):
        return FastLorenzGenerator(t_max=10.0, discard_transient=1.0, random_seed=0)

    def test_returns_1d_array(self, gen):
        traj = gen.generate_single_trajectory(node_id=0)
        assert traj.ndim == 1

    def test_positive_length(self, gen):
        traj = gen.generate_single_trajectory(node_id=0)
        assert len(traj) > 0

    def test_values_in_unit_interval(self, gen):
        traj = gen.generate_single_trajectory(node_id=0)
        assert np.all(traj >= 0.0)
        assert np.all(traj <= 1.0 + 1e-9)

    def test_values_finite(self, gen):
        traj = gen.generate_single_trajectory(node_id=0)
        assert np.all(np.isfinite(traj))

    def test_different_nodes_differ(self, gen):
        t0 = gen.generate_single_trajectory(node_id=0)
        t1 = gen.generate_single_trajectory(node_id=1)
        assert not np.allclose(t0, t1)

    def test_expected_length(self):
        gen = FastLorenzGenerator(t_max=5.0, discard_transient=1.0, dt=0.05, random_seed=1)
        traj = gen.generate_single_trajectory(node_id=0)
        expected_steps = int((5.0 - 1.0) / 0.05)
        assert abs(len(traj) - expected_steps) <= 2

    def test_zero_transient_not_discarded(self):
        gen = FastLorenzGenerator(t_max=2.0, discard_transient=0.0, dt=0.05, random_seed=0)
        traj = gen.generate_single_trajectory(node_id=0)
        expected = int(2.0 / 0.05)
        assert abs(len(traj) - expected) <= 2


# ---------------------------------------------------------------------------
# generate_ensemble
# ---------------------------------------------------------------------------

class TestGenerateEnsemble:
    @pytest.fixture
    def gen(self):
        return FastLorenzGenerator(t_max=5.0, discard_transient=1.0, random_seed=42)

    def test_shape(self, gen):
        arr = gen.generate_ensemble(n_nodes=3)
        assert arr.ndim == 2
        assert arr.shape[1] == 3

    def test_all_values_finite(self, gen):
        arr = gen.generate_ensemble(n_nodes=3)
        assert np.all(np.isfinite(arr))

    def test_all_values_normalized(self, gen):
        arr = gen.generate_ensemble(n_nodes=3)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0 + 1e-9)

    def test_columns_differ(self, gen):
        arr = gen.generate_ensemble(n_nodes=2)
        assert not np.allclose(arr[:, 0], arr[:, 1])

    def test_single_node(self, gen):
        arr = gen.generate_ensemble(n_nodes=1)
        assert arr.shape[1] == 1


# ---------------------------------------------------------------------------
# generate_ensemble_df (only if pandas available)
# ---------------------------------------------------------------------------

class TestGenerateEnsembleDf:
    def test_import_error_without_pandas(self, monkeypatch):
        import phase_omega.lorenz as lorenz_mod
        original = lorenz_mod._PANDAS_AVAILABLE
        monkeypatch.setattr(lorenz_mod, "_PANDAS_AVAILABLE", False)
        gen = FastLorenzGenerator(t_max=2.0, discard_transient=0.5)
        with pytest.raises(ImportError):
            gen.generate_ensemble_df(n_nodes=2)
        monkeypatch.setattr(lorenz_mod, "_PANDAS_AVAILABLE", original)

    def test_returns_dataframe_if_pandas_available(self):
        pytest.importorskip("pandas")
        import pandas as pd
        gen = FastLorenzGenerator(t_max=3.0, discard_transient=0.5, random_seed=0)
        df = gen.generate_ensemble_df(n_nodes=2)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 2
        assert list(df.columns) == ["node_0", "node_1"]

    def test_dataframe_column_names(self):
        pytest.importorskip("pandas")
        gen = FastLorenzGenerator(t_max=3.0, discard_transient=0.5, random_seed=0)
        df = gen.generate_ensemble_df(n_nodes=4)
        assert set(df.columns) == {"node_0", "node_1", "node_2", "node_3"}


# ---------------------------------------------------------------------------
# Lorenz attractor properties
# ---------------------------------------------------------------------------

class TestLorenzProperties:
    def test_trajectory_has_variation(self):
        gen = FastLorenzGenerator(t_max=20.0, discard_transient=2.0, random_seed=0)
        traj = gen.generate_single_trajectory(node_id=0)
        assert np.std(traj) > 0.05

    def test_not_constant(self):
        gen = FastLorenzGenerator(t_max=10.0, discard_transient=1.0, random_seed=0)
        traj = gen.generate_single_trajectory(node_id=0)
        assert np.max(traj) - np.min(traj) > 0.1
