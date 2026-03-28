"""Tests for ReturnTimeStatistics."""

import math

import numpy as np
import pytest

from omniagis.core.return_time import ReturnTimeStatistics


class TestFindReturns:
    def test_periodic_sine(self) -> None:
        """Sine wave hits 0 twice per period; returns should be ~evenly spaced."""
        t = np.linspace(0, 4 * np.pi, 1000, endpoint=False)
        series = np.sin(t)
        rts = ReturnTimeStatistics(tolerance=0.05)
        indices = rts.find_returns(series, target_value=0.0)
        # Sine crosses zero ~8 times in [0, 4π]; with tol we expect groups
        assert len(indices) > 0

    def test_exact_hits(self) -> None:
        series = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0], dtype=float)
        rts = ReturnTimeStatistics(tolerance=0.0)
        indices = rts.find_returns(series, target_value=0.0)
        np.testing.assert_array_equal(indices, [0, 3, 5])

    def test_tolerance_override(self) -> None:
        series = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=float)
        rts = ReturnTimeStatistics(tolerance=0.0)
        indices = rts.find_returns(series, target_value=0.0, tol=0.6)
        assert 1 in indices and 3 in indices

    def test_1d_required(self) -> None:
        rts = ReturnTimeStatistics()
        with pytest.raises(ValueError):
            rts.find_returns(np.zeros((5, 2)), 0.0)


class TestComputeStats:
    def test_known_gaps(self) -> None:
        # gaps = [2, 3, 5]  → mean=3.33, min=2, max=5
        indices = np.array([0, 2, 5, 10])
        rts = ReturnTimeStatistics()
        stats = rts.compute_stats(indices)
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(10 / 3, rel=1e-6)
        assert stats["min"] == pytest.approx(2.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_single_return(self) -> None:
        rts = ReturnTimeStatistics()
        stats = rts.compute_stats(np.array([7]))
        assert stats["count"] == 1
        assert math.isinf(stats["mean"])

    def test_empty(self) -> None:
        rts = ReturnTimeStatistics()
        stats = rts.compute_stats(np.array([], dtype=int))
        assert stats["count"] == 0
        assert math.isinf(stats["mean"])


class TestClassify:
    def test_pass(self) -> None:
        rts = ReturnTimeStatistics()
        stats = {"mean": 5.0, "count": 10}
        assert rts.classify(stats, max_allowed_mean=10.0) == "PASS"

    def test_partial_pass(self) -> None:
        rts = ReturnTimeStatistics()
        stats = {"mean": 15.0, "count": 10}
        assert rts.classify(stats, max_allowed_mean=10.0) == "PARTIAL PASS"

    def test_no_pass_high_mean(self) -> None:
        rts = ReturnTimeStatistics()
        stats = {"mean": 100.0, "count": 10}
        assert rts.classify(stats, max_allowed_mean=10.0) == "NO PASS"

    def test_no_pass_insufficient_returns(self) -> None:
        rts = ReturnTimeStatistics()
        stats = {"mean": 1.0, "count": 1}
        assert rts.classify(stats, max_allowed_mean=10.0) == "NO PASS"
