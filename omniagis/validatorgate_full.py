"""
validatorgate_full.py — core scientific validation engine.

Primary observable : return-time statistics tau_i = t_{i+1} - t_i
                     (times between consecutive visits to a target set U).

Validation pipeline
-------------------
1. compute_return_times  — vectorised inter-visit times
2. survival_function     — empirical S(n) = P(tau > n)  via searchsorted
3. fit_power_law_tail    — OLS in log-log space for n >= n_min only;
                           REJECTED if count(tau >= n_min) < min_tail_sample_size;
                           FILTERED to n where count(tau > n) >= min_tail_obs so that
                           only well-supported S(n) estimates enter the regression —
                           this makes the effective regression range scale with N and
                           is necessary for bootstrap CI widths to shrink correctly
4. detect_plateau        — sliding-window log-log slopes; np.isfinite guard on NaN
5. bootstrap_ci          — rng.choice(replace=True); np.isnan fits discarded
6. multi_scale_ci        — three scales N, N//2, N//4;
                           alpha_hat reported per scale (never averaged);
                           best CI = index 0 (max sample size)
7. validate              — fail-closed gate returning a structured dict with
                           all six checks: R2_ok, CI_width_ok, theory_in_CI,
                           CI_shrinks, plateau, tail_mass
                           verdict: "ACCEPTED" iff all checks pass, else "FAIL_CLOSED"

Reproducibility
---------------
Every random operation accepts an explicit numpy.random.Generator so that
results are fully deterministic given a seed.  No global random state is used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. Return-time computation
# ---------------------------------------------------------------------------

def compute_return_times(
    trajectory: NDArray[np.float64],
    in_target: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> NDArray[np.int64]:
    """Return successive inter-visit times for the target set U.

    Parameters
    ----------
    trajectory :
        1-D array of scalar observations x(0), x(1), ..., x(T-1).
    in_target :
        Vectorised boolean predicate applied to the whole trajectory at once.
        Example: ``lambda x: (x >= 0.0) & (x <= epsilon)``

    Returns
    -------
    tau : int64 array of length (#visits - 1).
        tau[i] = t_{i+1} - t_i (gap between consecutive target-set visits).
    """
    mask = np.asarray(in_target(trajectory), dtype=bool)
    hits = np.nonzero(mask)[0]
    if hits.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.diff(hits).astype(np.int64)


# ---------------------------------------------------------------------------
# 2. Survival function  S(n) = P(tau > n)
# ---------------------------------------------------------------------------

def survival_function(
    tau: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Empirical survival function S(n) = P(tau > n).

    Uses np.searchsorted on the sorted tau array for O(max_tau * log N) time.

    Returns
    -------
    ns : int64 array  1 .. max(tau)
    S  : float64 array  S[i] = fraction of tau strictly greater than ns[i]
    """
    if tau.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    tau_sorted = np.sort(tau)
    N = tau_sorted.size
    ns = np.arange(1, int(tau_sorted[-1]) + 1, dtype=np.int64)
    # count(tau > n) = N - (# elements <= n) = N - searchsorted(side='right')
    counts = N - np.searchsorted(tau_sorted, ns, side="right")
    S = counts.astype(np.float64) / N
    return ns, S


# ---------------------------------------------------------------------------
# 3. Tail-only power-law fitting  (OLS in log-log, n >= n_min)
# ---------------------------------------------------------------------------

@dataclass
class PowerLawFit:
    alpha: float           # exponent: S(n) ~ C * n^(-alpha)
    log_c: float           # log-space intercept
    r_squared: float       # R² of log-log OLS fit
    n_tail_points: int     # number of (n, S(n)) pairs used in regression
    tail_sample_size: int  # count(tau >= n_min)  [raw observations]
    n_min: int             # lower bound of fitting region
    min_tail_obs: int      # minimum observations backing each S(n) data point


def fit_power_law_tail(
    ns: NDArray[np.int64],
    S: NDArray[np.float64],
    tau: NDArray[np.int64],
    n_min: int,
    min_tail_sample_size: int = 1000,
    min_tail_obs: int = 10,
) -> PowerLawFit:
    """Fit S(n) ~ C * n^{-alpha} via OLS in log-log space for n >= n_min.

    Two rejection / filtering criteria apply:

    1. **Tail sample size gate** — If ``count(tau >= n_min) < min_tail_sample_size``
       the fit is rejected outright (all NaN).  This prevents spurious fits on
       sparse tails.

    2. **Minimum-observations filter** — Only (n, S(n)) pairs where
       ``count(tau > n) >= min_tail_obs`` are included in the OLS regression.
       This discards the noisy far tail (S(n) ≈ 0 backed by <min_tail_obs
       observations) whose variance would otherwise dominate the regression and
       prevent bootstrap CI widths from shrinking with N.  The effective
       regression range automatically scales with N.

    Parameters
    ----------
    ns, S :
        Output of :func:`survival_function`.
    tau :
        Raw return-time array used to count tail observations.
    n_min :
        Minimum n for the fitting region.
    min_tail_sample_size :
        Minimum required count(tau >= n_min).  Default 1000.
    min_tail_obs :
        Minimum ``count(tau > n)`` required for each regression point.
        Default 10.  Set to 0 to disable this filter.
    """
    tail_sample_size = int(np.sum(tau >= n_min))
    N = tau.size

    _nan = PowerLawFit(
        alpha=float("nan"),
        log_c=float("nan"),
        r_squared=float("nan"),
        n_tail_points=0,
        tail_sample_size=tail_sample_size,
        n_min=n_min,
        min_tail_obs=min_tail_obs,
    )

    if tail_sample_size < min_tail_sample_size:
        return _nan

    # Tail mask:
    #  - n >= n_min
    #  - S(n) > 0  (log is defined)
    #  - count(tau > n) = S(n) * N >= min_tail_obs  (sufficient observations)
    obs_counts = np.round(S * N).astype(np.int64)
    tail_mask = (ns >= n_min) & (S > 0.0) & (obs_counts >= min_tail_obs)
    ns_t = ns[tail_mask]
    S_t = S[tail_mask]

    if ns_t.size < 2:
        return _nan

    log_n = np.log(ns_t.astype(np.float64))
    log_S = np.log(S_t)

    # OLS:  log_S = log_c + slope * log_n,  slope = -alpha
    A = np.column_stack([np.ones(log_n.size), log_n])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_S, rcond=None)
    log_c = float(coeffs[0])
    alpha = float(-coeffs[1])          # slope is -alpha

    log_S_pred = log_c - alpha * log_n
    ss_res = float(np.sum((log_S - log_S_pred) ** 2))
    ss_tot = float(np.sum((log_S - log_S.mean()) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return PowerLawFit(
        alpha=alpha,
        log_c=log_c,
        r_squared=r_squared,
        n_tail_points=int(ns_t.size),
        tail_sample_size=tail_sample_size,
        n_min=n_min,
        min_tail_obs=min_tail_obs,
    )


# ---------------------------------------------------------------------------
# 4. Plateau detection  (log-log slopes, NaN-safe, sliding-window rel std/mean)
# ---------------------------------------------------------------------------

@dataclass
class PlateauResult:
    detected: bool
    rel_variation: float    # min std(slopes) / |mean(slopes)| over all windows
    plateau_start: int      # n-value at window start  (0 if none found)
    plateau_end: int        # n-value at window end    (0 if none found)
    n_valid_slopes: int     # finite slopes used in the scan


def detect_plateau(
    ns: NDArray[np.int64],
    S: NDArray[np.float64],
    window_fraction: float = 0.2,
    rel_threshold: float = 0.15,
) -> PlateauResult:
    """Detect a scale-invariant plateau in S(n) via log-log slope analysis.

    Algorithm
    ---------
    1. Restrict to positive-S points (log is defined).
    2. Compute finite-difference log-log slopes  d(log S)/d(log n).
    3. **Discard non-finite slopes** (NaN / ±inf) produced by S → 0 edges.
    4. Slide a window over valid slopes; compute std / |mean| per window.
    5. Plateau detected iff min(std / |mean|) < rel_threshold.

    Parameters
    ----------
    window_fraction :
        Window width as a fraction of the number of valid slopes.
    rel_threshold :
        Maximum std / |mean| for a plateau segment.
    """
    pos = S > 0.0
    ns_p = ns[pos].astype(np.float64)
    S_p = S[pos]

    if ns_p.size < 4:
        return PlateauResult(
            detected=False, rel_variation=float("nan"),
            plateau_start=0, plateau_end=0, n_valid_slopes=0,
        )

    log_n = np.log(ns_p)
    log_S = np.log(S_p)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_slopes = np.diff(log_S) / np.diff(log_n)

    # Discard non-finite slopes (NaN, ±inf)
    finite_mask = np.isfinite(raw_slopes)
    slopes = raw_slopes[finite_mask]
    n_valid_slopes = int(slopes.size)

    if n_valid_slopes < 4:
        return PlateauResult(
            detected=False, rel_variation=float("nan"),
            plateau_start=0, plateau_end=0, n_valid_slopes=n_valid_slopes,
        )

    window = max(2, int(window_fraction * n_valid_slopes))
    best_rel = float("inf")
    best_i = 0

    for i in range(n_valid_slopes - window + 1):
        seg = slopes[i : i + window]
        m = abs(float(np.mean(seg)))
        if m == 0.0:
            continue
        rel = float(np.std(seg) / m)
        if rel < best_rel:
            best_rel = rel
            best_i = i

    detected = math.isfinite(best_rel) and best_rel < rel_threshold

    if detected:
        # Map window position back to ns_p indices via finite_mask
        valid_indices = np.where(finite_mask)[0]
        w_start_idx = valid_indices[best_i]
        w_end_idx = valid_indices[min(best_i + window - 1, n_valid_slopes - 1)]
        plateau_start = int(ns_p[w_start_idx])
        plateau_end = int(ns_p[min(w_end_idx + 1, ns_p.size - 1)])
    else:
        plateau_start = plateau_end = 0

    return PlateauResult(
        detected=detected,
        rel_variation=float(best_rel) if math.isfinite(best_rel) else float("nan"),
        plateau_start=plateau_start,
        plateau_end=plateau_end,
        n_valid_slopes=n_valid_slopes,
    )


# ---------------------------------------------------------------------------
# 5. Bootstrap CI  (rng.choice replace=True; np.isnan discard)
# ---------------------------------------------------------------------------

def _alpha_from_tau(
    tau: NDArray[np.int64],
    n_min: int,
    min_tail_sample_size: int,
    min_tail_obs: int,
) -> float:
    """Fit power-law alpha from a return-time array.  Returns NaN on failure."""
    if tau.size < 2:
        return float("nan")
    ns, S = survival_function(tau)
    if ns.size == 0:
        return float("nan")
    fit = fit_power_law_tail(
        ns, S, tau,
        n_min=n_min,
        min_tail_sample_size=min_tail_sample_size,
        min_tail_obs=min_tail_obs,
    )
    return fit.alpha


def bootstrap_ci(
    tau: NDArray[np.int64],
    n_min: int,
    min_tail_sample_size: int = 1000,
    min_tail_obs: int = 10,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, int]:
    """Bootstrap CI for the power-law alpha.

    Each resample is drawn with ``rng.choice(tau, size=N, replace=True)``.
    Resamples that produce a NaN fit (sparse tail, OLS failure, etc.) are
    **discarded** via ``np.isnan``.  The CI is computed from valid fits only.

    The ``min_tail_obs`` filter is forwarded to :func:`fit_power_law_tail`
    so that each bootstrap fit uses only well-supported regression points,
    ensuring the effective regression range scales with the resample size.

    Returns
    -------
    (lower, upper, n_valid)
        CI bounds and the number of valid bootstrap alpha estimates used.
        lower = upper = NaN when fewer than 2 valid fits remain.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = tau.size
    boot_alphas: list[float] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(tau, size=N, replace=True)
        a = _alpha_from_tau(sample, n_min, min_tail_sample_size, min_tail_obs)
        if not np.isnan(a):
            boot_alphas.append(a)

    n_valid = len(boot_alphas)
    if n_valid < 2:
        return float("nan"), float("nan"), n_valid

    arr = np.array(boot_alphas, dtype=np.float64)
    p_lo = 100.0 * (1.0 - confidence) / 2.0
    p_hi = 100.0 - p_lo
    return (
        float(np.percentile(arr, p_lo)),
        float(np.percentile(arr, p_hi)),
        n_valid,
    )


# ---------------------------------------------------------------------------
# 6. Multi-scale CI  (N, N//2, N//4; per-scale alpha; best = index 0)
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleCI:
    scales: list[int]       # [N, N//2, N//4]
    lower: list[float]      # bootstrap CI lower bound per scale
    upper: list[float]      # bootstrap CI upper bound per scale
    width: list[float]      # CI width per scale
    alpha_hat: list[float]  # OLS alpha on each sub-sample (never averaged)
    n_valid_fits: list[int] # valid bootstrap fits per scale
    best_scale_idx: int     # index of max-sample-size CI (always 0)
    ci_shrinks: bool        # width[2] > width[1] > width[0]  (strict)


def multi_scale_ci(
    tau: NDArray[np.int64],
    n_min: int,
    min_tail_sample_size: int = 1000,
    min_tail_obs: int = 10,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> MultiScaleCI:
    """Bootstrap CIs at three scales: N (full), N//2, N//4.

    A single trajectory of N return times is split into prefix slices
    [0:N], [0:N//2], [0:N//4].  This is valid under stationarity (ergodic
    systems).  Within each scale the CI is obtained by bootstrap resampling
    with replacement.

    The ``min_tail_obs`` filter ensures the OLS regression at each scale uses
    only S(n) points backed by at least ``min_tail_obs`` observations.  This
    makes the effective regression range proportional to N, so that bootstrap
    CI widths decrease monotonically with N (CI shrink check).

    The **best CI** (used for CI_width_ok and theory_in_CI) is always at
    ``best_scale_idx = 0`` (maximum sample size = full N).

    Per-scale alpha_hat values are reported individually.
    Slopes are **never averaged** across scales.

    CI shrink criterion: width[N/4] > width[N/2] > width[N]  (strict).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = tau.size
    sizes = [N, max(4, N // 2), max(4, N // 4)]

    scales_out: list[int] = []
    lower_out: list[float] = []
    upper_out: list[float] = []
    width_out: list[float] = []
    alpha_hat_out: list[float] = []
    n_valid_out: list[int] = []

    for sz in sizes:
        sub = tau[:sz]

        # OLS alpha on this sub-sample (point estimate, not bootstrapped)
        ns_s, S_s = survival_function(sub)
        if ns_s.size > 0:
            fit_s = fit_power_law_tail(
                ns_s, S_s, sub,
                n_min=n_min,
                min_tail_sample_size=min_tail_sample_size,
                min_tail_obs=min_tail_obs,
            )
            a_hat = fit_s.alpha
        else:
            a_hat = float("nan")

        # Bootstrap CI on this sub-sample
        lo, hi, nv = bootstrap_ci(
            sub,
            n_min=n_min,
            min_tail_sample_size=min_tail_sample_size,
            min_tail_obs=min_tail_obs,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            rng=rng,
        )
        w = (hi - lo) if (math.isfinite(lo) and math.isfinite(hi)) else float("nan")

        scales_out.append(sz)
        lower_out.append(lo)
        upper_out.append(hi)
        width_out.append(w)
        alpha_hat_out.append(a_hat)
        n_valid_out.append(nv)

    # CI shrink check: widths must strictly decrease as sample size increases.
    # width[0] = largest sample size N   → narrowest CI
    # width[2] = smallest sample size N/4 → widest CI
    w0, w1, w2 = width_out[0], width_out[1], width_out[2]
    ci_shrinks = (
        math.isfinite(w0) and math.isfinite(w1) and math.isfinite(w2)
        and w2 > w1 > w0
    )

    return MultiScaleCI(
        scales=scales_out,
        lower=lower_out,
        upper=upper_out,
        width=width_out,
        alpha_hat=alpha_hat_out,
        n_valid_fits=n_valid_out,
        best_scale_idx=0,
        ci_shrinks=ci_shrinks,
    )


# ---------------------------------------------------------------------------
# 7. Validation configuration & fail-closed gate
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    # Tail fitting
    n_min: int = 5
    """Lower bound for the power-law fitting region (n >= n_min)."""

    min_tail_sample_size: int = 1000
    """Minimum count(tau >= n_min) required; fit rejected below this."""

    min_tail_obs: int = 10
    """Minimum count(tau > n) required for each (n, S(n)) regression point.
    Filters out the sparse far tail whose OLS contribution would prevent CI
    widths from shrinking with N.  Set to 0 to disable."""

    # Power-law quality
    min_r_squared: float = 0.80
    """Minimum R² (log-log OLS) to declare the fit acceptable."""

    alpha_min: float = 0.0
    """Exclusive lower bound: fitted alpha must satisfy alpha > alpha_min."""

    alpha_max: float = 10.0
    """Exclusive upper bound: fitted alpha must satisfy alpha < alpha_max."""

    # Theoretical reference  (optional)
    theory_alpha: Optional[float] = None
    """Expected exponent from theory (e.g. 1/(z-1) for PM map).
    If None, theory_in_CI is set to True (no check performed)."""

    # CI quality
    max_ci_rel_width: float = 0.5
    """Maximum acceptable CI width relative to |alpha_hat|."""

    # Plateau
    plateau_window_fraction: float = 0.2
    plateau_rel_threshold: float = 0.15
    require_plateau: bool = False
    """If False, plateau check passes unconditionally when not detected."""

    # Tail mass
    min_tail_mass: float = 0.01
    """Minimum fraction of return times with tau >= n_min."""

    # Bootstrap
    n_bootstrap: int = 500
    bootstrap_confidence: float = 0.95

    # Reproducibility
    seed: int = 42


def validate(
    trajectory: NDArray[np.float64],
    in_target: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
    config: Optional[ValidationConfig] = None,
) -> dict:
    """Run full fail-closed validation on a trajectory.

    Parameters
    ----------
    trajectory :
        1-D time series from the dynamical system.
    in_target :
        Vectorised boolean predicate for the target set U.
        Example: ``lambda x: (x >= 0.0) & (x <= epsilon)``
    config :
        Validation parameters (default: :class:`ValidationConfig`).

    Returns
    -------
    dict with keys:

    ``verdict``
        ``"ACCEPTED"`` iff **all** six checks pass; ``"FAIL_CLOSED"`` otherwise.
    ``n_return_times``
        Number of inter-visit times extracted from the trajectory.
    ``checks``
        Dict with six explicit boolean keys:
        ``R2_ok``, ``CI_width_ok``, ``theory_in_CI``,
        ``CI_shrinks``, ``plateau``, ``tail_mass``.
    ``fail_reasons``
        List of human-readable strings for every failed check.
    ``diagnostics``
        Nested dict with raw numeric results for each stage.
    """
    if config is None:
        config = ValidationConfig()

    rng = np.random.default_rng(config.seed)
    fail_reasons: list[str] = []

    # ------------------------------------------------------------------
    # Stage 1 — return times
    # ------------------------------------------------------------------
    tau = compute_return_times(trajectory, in_target)
    n_tau = int(tau.size)

    # ------------------------------------------------------------------
    # Stage 2 — survival function
    # ------------------------------------------------------------------
    ns, S = survival_function(tau)

    # ------------------------------------------------------------------
    # Stage 3 — power-law tail fit
    # ------------------------------------------------------------------
    if ns.size >= 2:
        pl = fit_power_law_tail(
            ns, S, tau,
            n_min=config.n_min,
            min_tail_sample_size=config.min_tail_sample_size,
            min_tail_obs=config.min_tail_obs,
        )
    else:
        pl = PowerLawFit(
            alpha=float("nan"), log_c=float("nan"),
            r_squared=float("nan"),
            n_tail_points=0,
            tail_sample_size=int(np.sum(tau >= config.n_min)) if n_tau > 0 else 0,
            n_min=config.n_min,
            min_tail_obs=config.min_tail_obs,
        )

    alpha_ok = (
        math.isfinite(pl.alpha)
        and config.alpha_min < pl.alpha < config.alpha_max
    )
    r2_ok = math.isfinite(pl.r_squared) and pl.r_squared >= config.min_r_squared
    R2_ok = alpha_ok and r2_ok

    if not R2_ok:
        if not math.isfinite(pl.alpha):
            fail_reasons.append(
                f"Power-law fit invalid: tail_sample_size={pl.tail_sample_size} "
                f"< min={config.min_tail_sample_size} or insufficient data"
            )
        elif not alpha_ok:
            fail_reasons.append(
                f"Power-law alpha={pl.alpha:.4f} outside "
                f"({config.alpha_min}, {config.alpha_max})"
            )
        if not r2_ok:
            fail_reasons.append(
                f"Power-law R²={pl.r_squared:.4f} < {config.min_r_squared}"
            )

    # ------------------------------------------------------------------
    # Stage 4 — plateau detection
    # ------------------------------------------------------------------
    plateau_res = detect_plateau(
        ns, S,
        window_fraction=config.plateau_window_fraction,
        rel_threshold=config.plateau_rel_threshold,
    )
    plateau_check = plateau_res.detected or not config.require_plateau
    if not plateau_check:
        fail_reasons.append("No scale-invariant plateau detected in S(n)")

    # ------------------------------------------------------------------
    # Stage 5 — tail mass
    # ------------------------------------------------------------------
    tail_mass_val = (
        float(np.sum(tau >= config.n_min)) / n_tau if n_tau > 0 else 0.0
    )
    tail_mass_ok = tail_mass_val >= config.min_tail_mass
    if not tail_mass_ok:
        fail_reasons.append(
            f"Tail mass {tail_mass_val:.4f} < min={config.min_tail_mass}"
        )

    # ------------------------------------------------------------------
    # Stage 6 — multi-scale bootstrap CI
    # ------------------------------------------------------------------
    if n_tau >= 4:
        ms = multi_scale_ci(
            tau,
            n_min=config.n_min,
            min_tail_sample_size=config.min_tail_sample_size,
            min_tail_obs=config.min_tail_obs,
            n_bootstrap=config.n_bootstrap,
            confidence=config.bootstrap_confidence,
            rng=rng,
        )
    else:
        ms = MultiScaleCI(
            scales=[n_tau, 0, 0],
            lower=[float("nan")] * 3,
            upper=[float("nan")] * 3,
            width=[float("nan")] * 3,
            alpha_hat=[float("nan")] * 3,
            n_valid_fits=[0] * 3,
            best_scale_idx=0,
            ci_shrinks=False,
        )

    # Best CI = index 0 (max sample size = full N)
    best_lo = ms.lower[0]
    best_hi = ms.upper[0]
    best_w = ms.width[0]
    best_alpha = ms.alpha_hat[0]

    CI_shrinks = ms.ci_shrinks
    if not CI_shrinks:
        fail_reasons.append(
            f"CI does not shrink: widths={[round(w, 4) if math.isfinite(w) else 'nan' for w in ms.width]}"
        )

    # CI_width_ok: relative width at best scale
    CI_width_ok = (
        math.isfinite(best_w)
        and math.isfinite(best_alpha)
        and best_alpha != 0.0
        and best_w / abs(best_alpha) < config.max_ci_rel_width
    )
    if not CI_width_ok:
        fail_reasons.append(
            (
                f"CI width too large: width={best_w:.4f}, "
                f"alpha={best_alpha:.4f}, "
                f"rel={best_w / abs(best_alpha):.4f} >= {config.max_ci_rel_width}"
            )
            if (math.isfinite(best_w) and math.isfinite(best_alpha) and best_alpha != 0.0)
            else "CI width check failed: invalid CI or alpha"
        )

    # theory_in_CI: trivially True when no theory value provided
    if config.theory_alpha is None:
        theory_in_CI = True
    else:
        theory_in_CI = (
            math.isfinite(best_lo)
            and math.isfinite(best_hi)
            and best_lo <= config.theory_alpha <= best_hi
        )
        if not theory_in_CI:
            fail_reasons.append(
                f"theory_alpha={config.theory_alpha} not in CI "
                f"[{best_lo:.4f}, {best_hi:.4f}]"
            )

    # ------------------------------------------------------------------
    # Fail-closed gate
    # ------------------------------------------------------------------
    checks: dict[str, bool] = {
        "R2_ok": R2_ok,
        "CI_width_ok": CI_width_ok,
        "theory_in_CI": theory_in_CI,
        "CI_shrinks": CI_shrinks,
        "plateau": plateau_check,
        "tail_mass": tail_mass_ok,
    }

    verdict = "ACCEPTED" if all(checks.values()) else "FAIL_CLOSED"

    return {
        "verdict": verdict,
        "n_return_times": n_tau,
        "checks": checks,
        "fail_reasons": fail_reasons,
        "diagnostics": {
            "power_law": asdict(pl),
            "plateau": asdict(plateau_res),
            "tail_mass": tail_mass_val,
            "multi_scale_ci": asdict(ms),
        },
    }
