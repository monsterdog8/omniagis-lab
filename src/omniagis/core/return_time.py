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
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = float(tolerance)

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
        effective_tol = self.tolerance if tol is None else float(tol)
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
        mean = stats.get("mean", float("inf"))
        count = stats.get("count", 0)

        if count < 2:
            return "NO PASS"

        if mean <= max_allowed_mean:
            return "PASS"
        elif mean <= 2.0 * max_allowed_mean:
            return "PARTIAL PASS"
        else:
            return "NO PASS"
