"""Tests for EpsilonRobustnessValidator."""

import numpy as np
import pytest

from omniagis.core.validator import EpsilonRobustnessValidator


EPSILON = 0.1


def make_validator() -> EpsilonRobustnessValidator:
    return EpsilonRobustnessValidator(epsilon=EPSILON)


class TestIdenticalTrajectories:
    def test_pass(self) -> None:
        v = make_validator()
        traj = np.ones((50, 3))
        ref = np.ones((50, 3))
        result = v.validate(traj, ref)
        assert result.status == "PASS"
        assert result.max_dist == pytest.approx(0.0)
        assert result.mean_dist == pytest.approx(0.0)
        assert result.epsilon == EPSILON


class TestSmallDeviation:
    def test_pass(self) -> None:
        v = make_validator()
        ref = np.zeros((100, 2))
        traj = ref + 0.05  # max dist = sqrt(0.05^2 + 0.05^2) ≈ 0.0707 < 0.1
        result = v.validate(traj, ref)
        assert result.status == "PASS"
        assert result.max_dist <= EPSILON


class TestLargeDeviation:
    def test_no_pass(self) -> None:
        v = make_validator()
        ref = np.zeros((100, 2))
        traj = ref + 1.0  # pointwise L2 ≈ 1.414 >> 0.1
        result = v.validate(traj, ref)
        assert result.status == "NO PASS"
        assert result.max_dist > EPSILON
        assert result.mean_dist > EPSILON


class TestModerateDeviation:
    def test_partial_pass(self) -> None:
        """max_dist > epsilon but mean_dist <= epsilon → PARTIAL PASS."""
        v = make_validator()
        ref = np.zeros((100, 1))
        traj = ref.copy()
        # One timestep has a large spike, rest are near-zero
        traj[0, 0] = 0.5   # max_dist = 0.5 > 0.1
        # mean ≈ 0.5/100 = 0.005 <= 0.1
        result = v.validate(traj, ref)
        assert result.status == "PARTIAL PASS"
        assert result.max_dist > EPSILON
        assert result.mean_dist <= EPSILON


class TestShapeMismatch:
    def test_raises(self) -> None:
        v = make_validator()
        with pytest.raises(ValueError):
            v.validate(np.zeros((10, 3)), np.zeros((10, 4)))


class TestVerdict:
    def test_verdict_alias(self) -> None:
        v = make_validator()
        traj = np.eye(5)
        ref = np.eye(5)
        assert v.verdict(traj, ref).status == "PASS"


class TestNegativeEpsilon:
    def test_raises(self) -> None:
        with pytest.raises(ValueError):
            EpsilonRobustnessValidator(epsilon=-1.0)
