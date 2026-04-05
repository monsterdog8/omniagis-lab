"""OMNIÆGIS core validation components.

GO OMNIÆGIS Core Module
========================

This module provides the fundamental scientific validation primitives for
dynamical systems analysis using fail-closed logic.

Components:
-----------
EpsilonRobustnessValidator:
    Validates trajectories [T, N] against reference trajectories using pointwise
    L2 distance metrics. Returns PASS if max(dist) ≤ ε, PARTIAL PASS if
    mean(dist) ≤ ε, else NO PASS.

ReturnTimeStatistics:
    Computes Poincaré recurrence statistics on 1-D scalar time series. Measures
    gaps between successive returns to a target value within tolerance. Classifies
    behavior based on mean return time vs. threshold.

FailClosedClassifier:
    Aggregates multiple verdicts using fail-closed logic:
        - Any "NO PASS" → "NO PASS"
        - Any "PARTIAL PASS" (and no "NO PASS") → "PARTIAL PASS"
        - All "PASS" → "PASS"

Usage:
------
    >>> from omniagis.core import EpsilonRobustnessValidator
    >>> import numpy as np
    >>> validator = EpsilonRobustnessValidator(epsilon=1e-3)
    >>> result = validator.validate(trajectory, reference)
    >>> print(result.status, result.max_dist, result.mean_dist)

    >>> from omniagis.core import ReturnTimeStatistics
    >>> rts = ReturnTimeStatistics(tolerance=0.05)
    >>> indices = rts.find_returns(series, target_value=0.0)
    >>> stats = rts.compute_stats(indices)
    >>> verdict = rts.classify(stats, max_allowed_mean=50)

    >>> from omniagis.core import FailClosedClassifier
    >>> clf = FailClosedClassifier()
    >>> global_verdict = clf.combine(["PASS", "PARTIAL PASS", "PASS"])
    >>> # → "PARTIAL PASS"
"""

from .validator import EpsilonRobustnessValidator
from .return_time import ReturnTimeStatistics
from .classifier import FailClosedClassifier

__all__ = ["EpsilonRobustnessValidator", "ReturnTimeStatistics", "FailClosedClassifier"]
