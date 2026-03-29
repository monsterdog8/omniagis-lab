"""ε-robustness validator for dynamical system trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationVerdict:
    """Result of an ε-robustness validation."""

    status: str          # "PASS" | "PARTIAL PASS" | "NO PASS"
    max_dist: float
    mean_dist: float
    epsilon: float


class EpsilonRobustnessValidator:
    """Validate a trajectory against a reference using pointwise L2 distance.

    Parameters
    ----------
    epsilon:
        Tolerance threshold.  All pointwise distances must be ≤ epsilon for
        a full PASS.  Only the mean distance must be ≤ epsilon for PARTIAL PASS.
    """

    def __init__(self, epsilon: float = 1e-3) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def validate(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
    ) -> ValidationVerdict:
        """Compare *trajectory* to *reference* pointwise.

        Parameters
        ----------
        trajectory:
            Array of shape ``[T, N]`` (T timesteps, N state dimensions).
        reference:
            Array of the same shape as *trajectory*.

        Returns
        -------
        ValidationVerdict
        """
        trajectory = np.asarray(trajectory, dtype=float)
        reference = np.asarray(reference, dtype=float)

        if trajectory.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch: trajectory {trajectory.shape} vs "
                f"reference {reference.shape}"
            )

        diff = trajectory - reference
        # pointwise L2 distances — shape [T]
        if diff.ndim == 1:
            distances = np.abs(diff)
        else:
            distances = np.sqrt(np.sum(diff**2, axis=1))

        max_dist = float(np.max(distances))
        mean_dist = float(np.mean(distances))

        if max_dist <= self.epsilon:
            status = "PASS"
        elif mean_dist <= self.epsilon:
            status = "PARTIAL PASS"
        else:
            status = "NO PASS"

        return ValidationVerdict(
            status=status,
            max_dist=max_dist,
            mean_dist=mean_dist,
            epsilon=self.epsilon,
        )

    # Convenience alias
    def verdict(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
    ) -> ValidationVerdict:
        """Alias for :meth:`validate`."""
        return self.validate(trajectory, reference)
