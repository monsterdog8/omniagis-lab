"""Coupled state-variable dynamics and fractal coherence tracking.

Two independent models:
  - ContinuumState / step_continuum: 6-variable discrete-time nonlinear dynamics
    (psi=coherence, F=fusion, S=entropy, C=chaos, R=resonance_hz, Q=energy)
  - FractalEngine / FractalState: cycle-based coherence tracking with history

Both are pure stdlib — no external dependencies.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Continuum dynamics (6-variable coupled system)
# ---------------------------------------------------------------------------

@dataclass
class ContinuumState:
    """6-variable coupled state for coherence/fusion/entropy/chaos/resonance/energy."""
    psi: float = 0.999    # coherence [0,1]
    F: float = 0.90       # fusion [0,1]
    S: float = 0.0001     # entropy [0,1]
    C: float = 0.15       # chaos [0,1]
    R: float = 55.435     # resonance Hz
    Q: float = 0.0        # energy (derived)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Hyperparameters (from source)
_A1, _B1 = 0.06, 0.02
_A2, _B2, _G2 = 0.05, 0.04, 0.015
_A3, _B3, _K3 = 0.03, 0.04, 0.02
_A4, _B4 = 0.03, 0.05
_D_COEFF = 0.9
_R_BASE = 55.435


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def energy(state: ContinuumState) -> float:
    """Q = (F * psi) / (1 + S) * R"""
    return (state.F * state.psi) / (1.0 + state.S) * state.R


def step_continuum(
    state: ContinuumState,
    *,
    noise_std: float = 0.002,
    rng: Optional[random.Random] = None,
) -> ContinuumState:
    """Advance the continuum state by one discrete time step with Gaussian noise.

    Equations (dt=1):
        psi' = psi + a1*(F - S) - b1*C + noise
        F'   = F   + a2*psi*(1-F) - b2*S*F + g2*C*(1-F) + noise
        S'   = S   + a3*C*(1-S) - b3*psi*S - k3*F*S + noise
        C'   = C   + a4*(1-C)*max(0, 1-psi) - b4*C*F + noise
        R    = R_base + d*(F - S)
        Q    = (F * psi) / (1 + S) * R
    """
    _rng = rng or random
    n = noise_std

    psi = state.psi
    F = state.F
    S = state.S
    C = state.C

    psi_next = psi + _A1 * (F - S) - _B1 * C + _rng.gauss(0.0, n)
    F_next = F + _A2 * psi * (1 - F) - _B2 * S * F + _G2 * C * (1 - F) + _rng.gauss(0.0, n)
    S_next = S + _A3 * C * (1 - S) - _B3 * psi * S - _K3 * F * S + _rng.gauss(0.0, n)
    C_next = C + _A4 * (1 - C) * max(0.0, 1.0 - psi) - _B4 * C * F + _rng.gauss(0.0, n)

    new_psi = _clamp01(psi_next)
    new_F = _clamp01(F_next)
    new_S = _clamp01(max(0.0, S_next))
    new_C = _clamp01(max(0.0, C_next))
    new_R = _R_BASE + _D_COEFF * (new_F - new_S)

    s = ContinuumState(psi=new_psi, F=new_F, S=new_S, C=new_C, R=new_R, Q=0.0)
    s.Q = energy(s)
    return s


def run_continuum(
    n_steps: int,
    initial: Optional[ContinuumState] = None,
    *,
    noise_std: float = 0.002,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run the continuum dynamics for n_steps. Returns list of state dicts."""
    rng = random.Random(seed)
    state = initial or ContinuumState()
    state.Q = energy(state)
    history = [state.to_dict()]
    for _ in range(n_steps):
        state = step_continuum(state, noise_std=noise_std, rng=rng)
        history.append(state.to_dict())
    return history


# ---------------------------------------------------------------------------
# Fractal coherence engine (cycle-based)
# ---------------------------------------------------------------------------

@dataclass
class FractalState:
    """Fractal coherence state.

    Attributes:
        coherence: Signal coherence ψ (target 1.0).
        entropy: 1 - coherence.
        resonance_hz: Current resonance frequency (Hz).
        drift: Deviation range from coherence history.
    """
    coherence: float = 1.0
    entropy: float = 0.0
    resonance_hz: float = 11.987
    drift: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


_FRACTAL_BASE_HZ = 11.987
_FRACTAL_BASE_COHERENCE = 0.99995
_FRACTAL_COHERENCE_MIN = 0.9999


class FractalEngine:
    """Cycle-based fractal coherence tracker.

    Computes coherence history with deterministic cycle-based drift (SHA-256)
    and time-based sine modulation. Pure stdlib — no external deps.
    """

    def __init__(self, history_size: int = 100) -> None:
        self.history_size = history_size
        self.coherence_history: Deque[float] = deque(maxlen=history_size)
        self.state = FractalState()
        self.cycle_count = 0

    def compute_metrics(self, cycle_id: Optional[int] = None) -> FractalState:
        """Compute fractal metrics for the next cycle step."""
        if cycle_id is None:
            cycle_id = self.cycle_count

        time_factor = (math.sin(time.time() * 0.01) + 1) / 2.0

        cycle_hash = hashlib.sha256(str(cycle_id).encode()).digest()
        cycle_drift = (cycle_hash[0] % 1000) / 500_000.0

        coherence = _FRACTAL_BASE_COHERENCE - cycle_drift + (time_factor * 0.00005)
        coherence = max(_FRACTAL_COHERENCE_MIN, min(1.0, coherence))

        self.coherence_history.append(coherence)

        entropy = 1.0 - coherence

        hist = list(self.coherence_history)
        if len(hist) > 10:
            mean_h = sum(hist) / len(hist)
            variance = sum((v - mean_h) ** 2 for v in hist) / len(hist)
        else:
            variance = 0.0
        resonance = _FRACTAL_BASE_HZ + variance * 1e6

        drift = (max(hist) - min(hist)) if len(hist) > 20 else 0.0

        self.state = FractalState(
            coherence=coherence,
            entropy=entropy,
            resonance_hz=resonance,
            drift=drift,
        )
        self.cycle_count += 1
        return self.state

    def get_state(self) -> FractalState:
        return self.state

    def get_history(self) -> List[float]:
        return list(self.coherence_history)

    def get_statistics(self) -> Dict[str, float]:
        """Return mean, std, min, max of coherence history."""
        hist = list(self.coherence_history)
        if not hist:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean_v = sum(hist) / len(hist)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in hist) / len(hist))
        return {
            "mean": float(mean_v),
            "std": float(std_v),
            "min": float(min(hist)),
            "max": float(max(hist)),
        }

    def is_stable(self, drift_threshold: float = 0.0001, coherence_threshold: float = 0.999) -> bool:
        """Return True if drift < threshold and coherence > coherence_threshold."""
        return self.state.drift < drift_threshold and self.state.coherence > coherence_threshold

    def reset(self) -> None:
        self.coherence_history.clear()
        self.state = FractalState()
        self.cycle_count = 0
