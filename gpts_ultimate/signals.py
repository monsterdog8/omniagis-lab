"""Time-series structure analysis — unified.

Combines gpts_core signal analysis (entropy, autocorr, AR2, spectral, motif)
with phase_omega world-agnostic metrics (memory, coherence, direction, spectral_entropy)
and null surrogates (phase_scramble, iid_shuffle, block_shuffle).

All functions available from one import.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core signal metrics (gpts_core origin)
# ---------------------------------------------------------------------------

def entropy_norm(x: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of histogram, normalized to [0, 1]."""
    counts, _ = np.histogram(x, bins=bins)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum() / math.log2(bins))


def compression_ratio(x: np.ndarray) -> float:
    """zlib compression ratio as a structure proxy (lower = more compressible = more structure)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < 1e-12:
        raw = bytes([0]) * len(x)
    else:
        raw = np.clip(((x - lo) / (hi - lo) * 255).astype(np.uint8), 0, 255).tobytes()
    return float(len(zlib.compress(raw, 9)) / len(raw))


def autocorr(x: np.ndarray, maxlag: int = 64) -> Tuple[float, float]:
    """Return (lag-1 autocorrelation, max absolute autocorrelation over 1..maxlag)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 1e-12:
        return 0.0, 0.0
    vals = [float(np.dot(x[:-lag], x[lag:]) / den) for lag in range(1, maxlag + 1)]
    return vals[0], float(np.max(np.abs(vals)))


def spectral_analysis(x: np.ndarray, sr: float) -> Tuple[float, float, float]:
    """Return (peak_freq_hz, spectral_concentration, spectral_entropy)."""
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 8 or np.std(x) < 1e-12:
        return 0.0, 0.0, 0.0
    y = np.fft.rfft(x * np.hanning(len(x)))
    power = np.abs(y) ** 2
    power[0] = 0.0
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0, 0.0, 0.0
    idx = int(np.argmax(power))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    conc = float(power[idx] / total)
    p = power[power > 0] / total
    ent = float(-(p * np.log2(p)).sum() / math.log2(len(power))) if len(power) > 1 else 0.0
    return float(freqs[idx]), conc, ent


def motif_share(x: np.ndarray, alphabet: int = 8, mlen: int = 4) -> float:
    """Fraction of the most common length-mlen ordinal motif."""
    qs = np.quantile(x, np.linspace(0, 1, alphabet + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    q = np.digitize(x, qs[1:-1]).astype(np.int64)
    if len(q) < mlen:
        return 0.0
    codes = q[:-3] * 512 + q[1:-2] * 64 + q[2:-1] * 8 + q[3:]
    counts = np.bincount(codes, minlength=alphabet ** mlen)
    total = int(counts.sum())
    return float(counts.max() / total) if total else 0.0


def window_stats(x: np.ndarray, sr: float, window: int = 512) -> Tuple[float, float, float]:
    """Return (peak_freq_cv, entropy_cv, segment_jsd_mean) across non-overlapping windows."""
    chunks = [x[i:i + window] for i in range(0, len(x) - window + 1, window)]
    if len(chunks) < 2:
        return float("nan"), float("nan"), float("nan")
    edges = np.histogram_bin_edges(x, bins=32)
    peaks, ents, hists = [], [], []
    for c in chunks:
        peaks.append(spectral_analysis(c, sr)[0])
        ents.append(entropy_norm(c))
        hist, _ = np.histogram(c, bins=edges)
        hist = hist.astype(float) + 1e-12
        hist /= hist.sum()
        hists.append(hist)
    pm = abs(float(np.mean(peaks)))
    pcv = float(np.std(peaks) / pm) if pm > 1e-9 else float("inf")
    em = abs(float(np.mean(ents)))
    ecv = float(np.std(ents) / em) if em > 1e-9 else float("inf")
    js = []
    for a, b in zip(hists[:-1], hists[1:]):
        m = 0.5 * (a + b)
        js.append(0.5 * (float(np.sum(a * np.log2(a / m))) + float(np.sum(b * np.log2(b / m)))))
    return pcv, ecv, float(np.mean(js))


def ar2_fit(x: np.ndarray, sr: float) -> Tuple[float, float, float, float, float]:
    """Fit AR(2) model. Return (a1, a2, r2, estimated_freq_hz, root_radius)."""
    y, x1, x2 = x[2:], x[1:-1], x[:-2]
    s11 = float(np.dot(x1, x1))
    s22 = float(np.dot(x2, x2))
    s12 = float(np.dot(x1, x2))
    b1 = float(np.dot(x1, y))
    b2 = float(np.dot(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-18:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    a1 = (b1 * s22 - b2 * s12) / det
    a2 = (s11 * b2 - s12 * b1) / det
    pred = a1 * x1 + a2 * x2
    resid = y - pred
    r2 = 1.0 - float(np.var(resid) / (np.var(y) + 1e-12))
    roots = np.roots([1.0, -a1, -a2])
    angles = [abs(np.angle(r)) for r in roots if abs(np.angle(r)) > 1e-9]
    freq = float(np.median(angles) * sr / (2 * math.pi)) if angles else 0.0
    rad = float(np.median([abs(r) for r in roots]))
    return float(a1), float(a2), float(r2), freq, rad


def structure_score(row: Dict[str, float]) -> float:
    """Composite structure score [0, 1] from metric dict produced by analyze()."""
    def c(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    ac = c((row["autocorr_max"] - 0.05) / 0.90)
    sc = c((row["spectral_concentration"] - 0.005) / 0.08)
    ar = c((row["ar2_r2"] - 0.05) / 0.90)
    mo = c((row["motif_top_share"] - 0.020) / 0.050)
    cr = c((1 - row["compression_ratio"]) / 0.15)
    pf = row["peak_freq_cv"]
    pfscore = 0.0 if not math.isfinite(pf) else c((0.50 - pf) / 0.50)
    js = row["segment_jsd_mean"]
    jscore = 0.0 if not math.isfinite(js) else c((0.20 - js) / 0.20)
    return 0.25 * ac + 0.20 * sc + 0.20 * ar + 0.10 * mo + 0.10 * cr + 0.10 * pfscore + 0.05 * jscore


def _norm(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-12)


def analyze(x: Sequence[float], sr: float = 256.0) -> Dict[str, float]:
    """Compute all structure metrics for a 1-D signal. Returns a flat dict."""
    arr = _norm(np.asarray(x, float))
    lag1, acmax = autocorr(arr)
    pf, spec_conc, spec_ent = spectral_analysis(arr, sr)
    pcv, ecv, jsd = window_stats(arr, sr)
    a1, a2, r2, ef, rr = ar2_fit(arr, sr)
    row: Dict[str, float] = {
        "entropy_norm": entropy_norm(arr),
        "compression_ratio": compression_ratio(arr),
        "autocorr_lag1": lag1,
        "autocorr_max": acmax,
        "spectral_peak_hz": pf,
        "spectral_concentration": spec_conc,
        "spectral_entropy": spec_ent,
        "motif_top_share": motif_share(arr),
        "peak_freq_cv": pcv,
        "entropy_cv": ecv,
        "segment_jsd_mean": jsd,
        "ar2_a1": a1,
        "ar2_a2": a2,
        "ar2_r2": r2,
        "estimated_freq_hz": ef,
        "root_radius": rr,
    }
    row["structure_score"] = structure_score(row)
    return row


def step_metrics(
    t: np.ndarray,
    y: np.ndarray,
    eps: float = 0.02,
) -> Dict[str, float]:
    """Compute step-response metrics: slope, rise time, overshoot, settling time, steady state."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 2:
        return {}
    y_final = float(np.mean(y[int(0.8 * n):]))

    n5 = max(2, int(0.05 * n))
    t5, y5 = t[:n5], y[:n5]
    slope = float(np.polyfit(t5, y5, 1)[0]) if len(t5) >= 2 else float("nan")

    lo10, hi90 = 0.1 * y_final, 0.9 * y_final
    cross10 = next((t[i] for i in range(n) if y[i] >= lo10), float("nan"))
    cross90 = next((t[i] for i in range(n) if y[i] >= hi90), float("nan"))
    tr = cross90 - cross10 if math.isfinite(cross10) and math.isfinite(cross90) else float("nan")

    y_peak = float(np.max(y))
    mp = (y_peak - y_final) / abs(y_final) if abs(y_final) > 1e-12 else float("nan")

    ts = float("nan")
    band = eps * abs(y_final)
    for i in range(n - 1, -1, -1):
        if abs(y[i] - y_final) > band:
            ts = float(t[i])
            break

    return {
        "initial_slope": slope,
        "rise_time": tr,
        "overshoot": mp,
        "settling_time": ts,
        "steady_state": y_final,
    }


def generate_null(x: np.ndarray, seed: int, kind: str) -> np.ndarray:
    """Generate a null-model copy of x.

    kind: shuffle | gaussian | same_marginal | phase_scramble
    """
    rng = np.random.default_rng(seed + 10_000_000)
    if kind == "shuffle":
        s = np.array(x)
        rng.shuffle(s)
        return s
    if kind == "gaussian":
        return rng.normal(float(np.mean(x)), float(np.std(x) + 1e-12), len(x))
    if kind == "same_marginal":
        return rng.choice(x, size=len(x), replace=True)
    if kind == "phase_scramble":
        return phase_scramble(x, seed=seed + 20_000_000)
    raise ValueError(f"Unknown null kind: {kind!r}")


def bootstrap_ci(
    values: List[float],
    seed: int,
    n_boot: int = 200,
) -> Dict[str, float]:
    """Bootstrap mean and 95% CI for a list of finite values."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    arr = np.asarray(finite, float)
    rng = np.random.default_rng(seed)
    bs = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(bs, 0.025)),
        "ci_high": float(np.quantile(bs, 0.975)),
        "n": len(finite),
    }


# ---------------------------------------------------------------------------
# World-agnostic shortcuts (phase_omega origin)
# ---------------------------------------------------------------------------

def memory(s: np.ndarray, max_lag: int = 1) -> float:
    """Temporal memory via abs lag-1 autocorrelation. High = predictable."""
    if len(s) < max_lag + 2:
        return 0.0
    s0, s1 = s[:-max_lag], s[max_lag:]
    if s0.std() < 1e-9 or s1.std() < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(s0, s1)[0, 1]))


def coherence(channels: np.ndarray) -> float:
    """Mean pairwise absolute correlation. channels: 2D (T, n_channels)."""
    if channels.ndim == 1 or channels.shape[1] < 2:
        return 0.0
    Cm = np.corrcoef(channels.T)
    iu = np.triu_indices(channels.shape[1], k=1)
    return float(np.nanmean(np.abs(Cm[iu])))


def direction(s: np.ndarray, window: int = 1) -> float:
    """Directional persistence: |sum(diff)| / sum(|diff|). High = trending."""
    d = np.diff(s)
    denom = float(np.sum(np.abs(d)))
    return float(abs(float(np.sum(d)))) / denom if denom > 1e-9 else 0.0


def spectral_entropy_agnostic(s: np.ndarray) -> float:
    """Shannon entropy of normalized power spectrum. High = noisy; Low = oscillatory."""
    s_centered = s - np.mean(s)
    ps = np.abs(np.fft.rfft(s_centered)) ** 2
    total = np.sum(ps)
    if total < 1e-12:
        return 0.0
    ps = ps / total
    ps = ps[ps > 0]
    return float(-np.sum(ps * np.log2(ps + 1e-12)))


# ---------------------------------------------------------------------------
# Null surrogates (phase_omega origin)
# ---------------------------------------------------------------------------

def phase_scramble(s: np.ndarray, seed: int = None) -> np.ndarray:
    """Destroy temporal structure, preserve spectral envelope (FFT phase randomization)."""
    if seed is not None:
        np.random.seed(seed)
    fft_vals = np.fft.rfft(s)
    phases = np.angle(fft_vals)
    magnitudes = np.abs(fft_vals)
    new_phases = np.random.uniform(-np.pi, np.pi, len(phases))
    new_phases[0] = phases[0]
    if len(phases) > 1:
        new_phases[-1] = phases[-1]
    return np.fft.irfft(magnitudes * np.exp(1j * new_phases), n=len(s)).real


def iid_shuffle(s: np.ndarray, seed: int = None) -> np.ndarray:
    """Complete permutation — destroys all temporal structure."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(s)


def block_shuffle(s: np.ndarray, block_size: int = 100, seed: int = None) -> np.ndarray:
    """Shuffle non-overlapping blocks — preserves local autocorrelation."""
    if seed is not None:
        np.random.seed(seed)
    n = len(s)
    n_blocks = n // block_size
    blocks = [s[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    remainder = s[n_blocks * block_size:]
    np.random.shuffle(blocks)
    parts = blocks + ([remainder] if len(remainder) > 0 else [])
    return np.concatenate(parts)
