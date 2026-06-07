"""Tests for MODULE_014_DYNAMICS.

Extracted from tests/test_gpts_core.py::TestDynamics.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestDynamics:
    def test_continuum_state_creation(self):
        from dynamics import ContinuumState
        s = ContinuumState(psi=0.8, F=0.7, S=0.3, C=0.2, R=40.0, Q=0.0)
        assert s.psi == 0.8
        assert s.F == 0.7

    def test_step_continuum_clamped(self):
        from dynamics import ContinuumState, step_continuum
        s = ContinuumState(psi=0.8, F=0.7, S=0.3, C=0.2, R=40.0, Q=0.0)
        s2 = step_continuum(s, noise_std=0.0)
        assert 0.0 <= s2.psi <= 1.0
        assert 0.0 <= s2.F <= 1.0
        assert 0.0 <= s2.S <= 1.0
        assert s2.Q >= 0.0

    def test_step_continuum_deterministic(self):
        from dynamics import ContinuumState, step_continuum
        s = ContinuumState(psi=0.5, F=0.5, S=0.5, C=0.5, R=20.0, Q=0.0)
        s2a = step_continuum(s, noise_std=0.0)
        s2b = step_continuum(s, noise_std=0.0)
        assert s2a.psi == s2b.psi

    def test_continuum_energy(self):
        from dynamics import ContinuumState, energy
        s = ContinuumState(psi=0.8, F=0.6, S=0.2, C=0.1, R=40.0, Q=0.0)
        q = energy(s)
        assert q >= 0.0
        expected = (s.F * s.psi) / (1 + s.S) * s.R
        assert abs(q - expected) < 1e-9

    def test_fractal_state_creation(self):
        from dynamics import FractalState
        s = FractalState(coherence=0.9999, entropy=0.0001, resonance_hz=12.0, drift=0.0)
        assert s.coherence == 0.9999

    def test_fractal_engine_compute(self):
        from dynamics import FractalEngine
        engine = FractalEngine()
        state = engine.compute_metrics(cycle_id=1)
        assert hasattr(state, "coherence")
        assert 0.0 <= state.coherence <= 1.0
        assert state.entropy >= 0.0

    def test_fractal_engine_history(self):
        from dynamics import FractalEngine
        engine = FractalEngine(history_size=20)
        for i in range(15):
            engine.compute_metrics(cycle_id=i)
        hist = engine.get_history()
        assert len(hist) == 15
        assert all(0.0 <= v <= 1.0 for v in hist)

    def test_fractal_engine_statistics(self):
        from dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(20):
            engine.compute_metrics(cycle_id=i)
        stats = engine.get_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    # Additional coverage tests

    def test_continuum_state_defaults(self):
        """Default ContinuumState values match documented constants."""
        from dynamics import ContinuumState
        s = ContinuumState()
        assert s.psi == 0.999
        assert s.F == 0.90
        assert abs(s.S - 0.0001) < 1e-12
        assert s.C == 0.15
        assert s.R == 55.435
        assert s.Q == 0.0

    def test_continuum_state_to_dict(self):
        """to_dict returns dict with all 6 fields."""
        from dynamics import ContinuumState
        s = ContinuumState()
        d = s.to_dict()
        assert isinstance(d, dict)
        for key in ("psi", "F", "S", "C", "R", "Q"):
            assert key in d

    def test_step_continuum_no_noise_energy_positive(self):
        """With zero noise, energy Q >= 0."""
        from dynamics import ContinuumState, step_continuum
        s = ContinuumState(psi=0.9, F=0.8, S=0.05, C=0.1, R=50.0, Q=0.0)
        s2 = step_continuum(s, noise_std=0.0)
        assert s2.Q >= 0.0

    def test_run_continuum_length(self):
        """run_continuum returns n_steps+1 entries."""
        from dynamics import run_continuum
        history = run_continuum(n_steps=10, seed=0)
        assert len(history) == 11

    def test_run_continuum_seeded_deterministic(self):
        """Same seed produces identical trajectories."""
        from dynamics import run_continuum
        h1 = run_continuum(n_steps=5, seed=7)
        h2 = run_continuum(n_steps=5, seed=7)
        assert h1 == h2

    def test_run_continuum_dict_keys(self):
        """Each state dict contains all 6 field keys."""
        from dynamics import run_continuum
        history = run_continuum(n_steps=3, seed=0)
        for entry in history:
            for key in ("psi", "F", "S", "C", "R", "Q"):
                assert key in entry

    def test_fractal_state_defaults(self):
        """Default FractalState values match documented constants."""
        from dynamics import FractalState
        s = FractalState()
        assert s.coherence == 1.0
        assert s.entropy == 0.0
        assert s.resonance_hz == 11.987
        assert s.drift == 0.0

    def test_fractal_state_to_dict(self):
        """to_dict returns dict with all 4 fields."""
        from dynamics import FractalState
        s = FractalState()
        d = s.to_dict()
        assert isinstance(d, dict)
        for key in ("coherence", "entropy", "resonance_hz", "drift"):
            assert key in d

    def test_fractal_engine_cycle_count_advances(self):
        """cycle_count advances by 1 on each compute_metrics call."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        assert engine.cycle_count == 0
        engine.compute_metrics()
        assert engine.cycle_count == 1
        engine.compute_metrics()
        assert engine.cycle_count == 2

    def test_fractal_engine_get_state_matches_last_compute(self):
        """get_state returns the state produced by the last compute_metrics."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        returned = engine.compute_metrics(cycle_id=42)
        got = engine.get_state()
        assert returned.coherence == got.coherence
        assert returned.entropy == got.entropy

    def test_fractal_engine_history_size_cap(self):
        """History deque respects history_size cap."""
        from dynamics import FractalEngine
        engine = FractalEngine(history_size=5)
        for i in range(10):
            engine.compute_metrics(cycle_id=i)
        hist = engine.get_history()
        assert len(hist) == 5

    def test_fractal_engine_statistics_empty(self):
        """get_statistics returns zeros when history is empty."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        stats = engine.get_statistics()
        assert stats == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    def test_fractal_engine_is_stable_after_reset(self):
        """After reset, drift=0, coherence=1.0, is_stable returns True."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(30):
            engine.compute_metrics(cycle_id=i)
        engine.reset()
        # Default state: coherence=1.0, drift=0.0
        assert engine.is_stable()

    def test_fractal_engine_reset_clears_history(self):
        """reset() clears history and resets cycle_count."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(10):
            engine.compute_metrics(cycle_id=i)
        engine.reset()
        assert len(engine.get_history()) == 0
        assert engine.cycle_count == 0

    def test_fractal_engine_entropy_equals_one_minus_coherence(self):
        """entropy = 1 - coherence for every computed cycle."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(5):
            state = engine.compute_metrics(cycle_id=i)
            assert abs(state.entropy - (1.0 - state.coherence)) < 1e-12

    def test_fractal_engine_coherence_in_valid_range(self):
        """Coherence is always in [0.9999, 1.0]."""
        from dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(50):
            state = engine.compute_metrics(cycle_id=i)
            assert 0.9999 <= state.coherence <= 1.0
