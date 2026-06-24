"""
BERZERKER World-Agnostic Metrics

Removes logistic-specific dependencies. Works on any time series.
Responds to CH-53: signal is world-agnostic, not logistic-specific.

CLAIM CEILING: LOCAL_SIMULATION_ONLY
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


class WorldAgnosticMetrics:
    """Emergent observers on any time series trajectory."""

    @staticmethod
    def memory(s: np.ndarray, max_lag: int = 1) -> float:
        """Temporal memory: abs autocorrelation at lag-1."""
        if len(s) < max_lag + 2:
            return 0.0
        s0, s1 = s[:-max_lag], s[max_lag:]
        if s0.std() < 1e-9 or s1.std() < 1e-9:
            return 0.0
        return float(abs(np.corrcoef(s0, s1)[0, 1]))

    @staticmethod
    def coherence(channels: np.ndarray) -> float:
        """Synchrony: mean pairwise absolute correlation across channels."""
        if channels.ndim == 1 or channels.shape[1] < 2:
            return 0.0
        Cm = np.corrcoef(channels.T)
        iu = np.triu_indices(channels.shape[1], k=1)
        vals = Cm[iu]
        return float(np.nanmean(np.abs(vals)))

    @staticmethod
    def direction(s: np.ndarray, window: int = 1) -> float:
        """Directional persistence: fraction of signed drift."""
        d = np.diff(s)
        denom = float(np.sum(np.abs(d)))
        return float(abs(float(np.sum(d)))) / denom if denom > 1e-9 else 0.0

    @staticmethod
    def spectral_entropy(s: np.ndarray) -> float:
        """Shannon entropy of normalized power spectrum."""
        s_centered = s - np.mean(s)
        ps = np.abs(np.fft.rfft(s_centered)) ** 2
        total = np.sum(ps)
        if total < 1e-12:
            return 0.0
        ps = ps / total
        ps = ps[ps > 0]
        return float(-np.sum(ps * np.log2(ps + 1e-12)))


class WorldAgnosticNulls:
    """Surrogate time series for null hypothesis testing."""

    @staticmethod
    def phase_scramble(trajectory: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Destroy temporal structure, preserve spectral envelope."""
        if seed is not None:
            np.random.seed(seed)
        fft_vals = np.fft.rfft(trajectory)
        phases = np.angle(fft_vals)
        magnitudes = np.abs(fft_vals)
        new_phases = np.random.uniform(-np.pi, np.pi, len(phases))
        new_phases[0] = phases[0]
        if len(phases) > 1:
            new_phases[-1] = phases[-1]
        new_fft = magnitudes * np.exp(1j * new_phases)
        null = np.fft.irfft(new_fft, n=len(trajectory))
        return null.real

    @staticmethod
    def iid_shuffle(trajectory: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Destroy all temporal structure: permute time points."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.permutation(trajectory)

    @staticmethod
    def block_shuffle(
        trajectory: np.ndarray, block_size: int = 100, seed: Optional[int] = None
    ) -> np.ndarray:
        """Destroy long-range structure, preserve local autocorrelation."""
        if seed is not None:
            np.random.seed(seed)
        n = len(trajectory)
        n_blocks = n // block_size
        blocks = [trajectory[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
        remainder = trajectory[n_blocks * block_size:]
        np.random.shuffle(blocks)
        parts = blocks + ([remainder] if len(remainder) > 0 else [])
        return np.concatenate(parts)


def compute_metrics(s: np.ndarray, all_channels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute world-agnostic fingerprint for a single time series."""
    if np.any(np.isnan(s)) or np.any(np.isinf(s)):
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    channels = all_channels if all_channels is not None else s.reshape(-1, 1)

    return {
        "memory": WorldAgnosticMetrics.memory(s),
        "coherence": WorldAgnosticMetrics.coherence(channels),
        "direction": WorldAgnosticMetrics.direction(s),
        "spectral_entropy": WorldAgnosticMetrics.spectral_entropy(s),
    }


def tribunal_verdict(metrics_real: Dict[str, float], null_dist: Dict) -> Dict:
    """Verdict logic for world-agnostic metrics.

    Verdicts: REFUTED_VS_NULL | OPEN_PROBLEM | EMPIRICAL_SIGNAL | PASS_LOCAL
    """
    signal_count = 0
    delta_d_sum = 0
    signals = []

    for metric_name, metric_value in metrics_real.items():
        null_info = null_dist.get(metric_name, {})
        if not null_info:
            continue
        null_p95 = null_info.get("p95", 0.0)
        if metric_value > null_p95:
            signal_count += 1
            delta_d_sum += 1
            signals.append({
                "metric": metric_name,
                "value": float(metric_value),
                "null_p95": float(null_p95),
                "excess": float(metric_value - null_p95),
            })

    if delta_d_sum == 0:
        verdict = "REFUTED_VS_NULL"
    elif signal_count <= 1:
        verdict = "OPEN_PROBLEM"
    elif delta_d_sum < 4:
        verdict = "EMPIRICAL_SIGNAL"
    else:
        verdict = "PASS_LOCAL"

    return {
        "verdict": verdict,
        "total_delta_d": delta_d_sum,
        "distinct_signatures": signal_count,
        "signals": signals,
        "production_unlocked": False,
        "claim_ceiling": "LOCAL_SIMULATION_ONLY",
    }
