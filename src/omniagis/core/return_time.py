"""Return-time (Poincaré recurrence) statistics for scalar time series."""

from __future__ import annotations

from typing import Dict

import numpy as np


class ReturnTimeStatistics:
    """Compute return-time statistics for a scalar time series.

    Parameters
    ----------
    tolerance:
        Maximum absolute deviation from *target_value* for a timestep to
        count as a "return".
    """

    def __init__(self, tolerance: float = 0.05) -> None:
        tolerance = float(tolerance)
        if not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def find_returns(
        self,
        series: np.ndarray,
        target_value: float,
        tol: float | None = None,
    ) -> np.ndarray:
        """Return indices where ``|series[t] - target_value| <= tol``.

        Parameters
        ----------
        series:
            1-D scalar time series.
        target_value:
            The value whose recurrences we are looking for.
        tol:
            Override instance tolerance for this call.

        Returns
        -------
        np.ndarray of int
            Array of timestep indices.
        """
        series = np.asarray(series, dtype=float)
        if series.ndim != 1:
            raise ValueError("series must be 1-D")
        if not np.all(np.isfinite(series)):
            raise ValueError("series must contain only finite values")
        target_value = float(target_value)
        if not np.isfinite(target_value):
            raise ValueError("target_value must be finite")
        effective_tol = self.tolerance if tol is None else float(tol)
        if not np.isfinite(effective_tol):
            raise ValueError("tol must be finite")
        if effective_tol < 0:
            raise ValueError("tol must be non-negative")
        (indices,) = np.where(np.abs(series - target_value) <= effective_tol)
        return indices.astype(int)

    def compute_stats(self, return_indices: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics of return times (gaps between returns).

        Parameters
        ----------
        return_indices:
            Array of timestep indices returned by :meth:`find_returns`.

        Returns
        -------
        dict with keys: mean, std, min, max, count
        """
        return_indices = np.asarray(return_indices, dtype=int)
        count = int(len(return_indices))

        if count < 2:
            return {
                "mean": float("inf"),
                "std": float("nan"),
                "min": float("inf"),
                "max": float("inf"),
                "count": count,
            }

        gaps = np.diff(return_indices).astype(float)
        return {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps)),
            "count": count,
        }

    def classify(
        self,
        stats: Dict[str, float],
        max_allowed_mean: float,
    ) -> str:
        """Classify return-time statistics.

        Returns
        -------
        "PASS" | "PARTIAL PASS" | "NO PASS"
        """
        mean = float(stats.get("mean", float("inf")))
        count = stats.get("count", 0)
        max_allowed_mean = float(max_allowed_mean)

        if not np.isfinite(max_allowed_mean) or max_allowed_mean < 0:
            raise ValueError("max_allowed_mean must be finite and non-negative")
        if count < 2 or not np.isfinite(mean):
            return "NO PASS"

        if mean <= max_allowed_mean:
            return "PASS"
        elif mean <= 2.0 * max_allowed_mean:
            return "PARTIAL PASS"
        else:
            return "NO PASS"
