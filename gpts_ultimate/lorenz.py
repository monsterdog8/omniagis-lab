"""
Ultra-Fast Lorenz Generator — Lightweight RK4

Replaces scipy.integrate.solve_ivp with direct RK4.
For BERZERKER Phase Omega: world-agnostic Lorenz trajectory generation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


class FastLorenzGenerator:
    """
    Ultra-fast Lorenz using direct RK4 integration.
    Generates normalized x-coordinate trajectories.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        dt: float = 0.05,
        t_max: float = 500.0,
        discard_transient: float = 50.0,
        random_seed: int = 42,
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.t_max = t_max
        self.discard_transient = discard_transient
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Single RK4 integration step for the Lorenz system."""
        x, y, z = state

        k1_x = self.sigma * (y - x)
        k1_y = x * (self.rho - z) - y
        k1_z = x * y - self.beta * z

        x2 = x + 0.5 * dt * k1_x
        y2 = y + 0.5 * dt * k1_y
        z2 = z + 0.5 * dt * k1_z
        k2_x = self.sigma * (y2 - x2)
        k2_y = x2 * (self.rho - z2) - y2
        k2_z = x2 * y2 - self.beta * z2

        x3 = x + 0.5 * dt * k2_x
        y3 = y + 0.5 * dt * k2_y
        z3 = z + 0.5 * dt * k2_z
        k3_x = self.sigma * (y3 - x3)
        k3_y = x3 * (self.rho - z3) - y3
        k3_z = x3 * y3 - self.beta * z3

        x4 = x + dt * k3_x
        y4 = y + dt * k3_y
        z4 = z + dt * k3_z
        k4_x = self.sigma * (y4 - x4)
        k4_y = x4 * (self.rho - z4) - y4
        k4_z = x4 * y4 - self.beta * z4

        return np.array([
            x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x),
            y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y),
            z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z),
        ])

    def generate_single_trajectory(self, node_id: int) -> np.ndarray:
        """
        Generate single Lorenz trajectory using RK4.
        Returns normalized x-coordinate time series in [0, 1].
        """
        state = np.random.uniform(-10, 10, 3)

        n_steps = int(self.t_max / self.dt)
        discard_steps = int(self.discard_transient / self.dt)

        trajectory: List[float] = []
        for step in range(n_steps):
            state = self.rk4_step(state, self.dt)
            if step >= discard_steps:
                trajectory.append(float(state[0]))

        if not trajectory:
            raise ValueError("No samples collected after transient discard")

        s = np.array(trajectory)
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min + 1e-10)

    def generate_ensemble(self, n_nodes: int = 20) -> np.ndarray:
        """
        Generate ensemble of n_nodes Lorenz trajectories.
        Returns 2D array of shape (T, n_nodes).
        """
        trajectories = [self.generate_single_trajectory(i) for i in range(n_nodes)]
        return np.column_stack(trajectories)

    def generate_ensemble_df(self, n_nodes: int = 20):
        """
        Generate ensemble as a pandas DataFrame (if pandas is available).
        Returns DataFrame with columns node_0, node_1, ..., node_{n_nodes-1}.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for generate_ensemble_df")
        data = self.generate_ensemble(n_nodes)
        return pd.DataFrame(data, columns=[f"node_{i}" for i in range(n_nodes)])
