"""Ulam spectral gap analysis for Pomeau-Manneville intermittency maps.

Discretizes the type-I PM map into a row-stochastic transition matrix (Ulam method),
then computes the canonical spectral gap Δ = 1 - |λ₂| and fits the log-log scaling
law log Δ = c + κ·log α to recover the universality exponent κ.

Requires: numpy

All results: LOCAL_LAB_ONLY — not external proof.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as _e:
    raise ImportError("spectral_gap requires numpy: pip install numpy") from _e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_PM: float = 0.75
KAPPA_THEORY: float = 2.0 / (2.0 - BETA_PM)   # = 1.6̄ for β=0.75
MIN_GAP: float = 5e-5
N_VALID_MIN: int = 6
R2_GOOD: float = 0.97
R2_PUB: float = 0.985
RMS_GOOD: float = 0.15
STDERR_PUB: float = 0.05
ROW_STOCH_TOL: float = 1e-8
LAMBDA1_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# Pomeau-Manneville map
# ---------------------------------------------------------------------------

def pm_map(x: np.ndarray, alpha: float, beta: float = BETA_PM) -> np.ndarray:
    """Vectorized Pomeau-Manneville type-I intermittency map.

    Left branch  (x < 0.5): x -> x + alpha * x^(1+beta)  mod 1
    Right branch (x ≥ 0.5): x -> 2x - 1
    """
    result = np.where(
        x < 0.5,
        np.clip(x + alpha * x ** (1.0 + beta), 0.0, 1.0 - 1e-15),
        np.clip(2.0 * x - 1.0, 0.0, 1.0 - 1e-15),
    )
    return result


# ---------------------------------------------------------------------------
# Ulam transition matrix
# ---------------------------------------------------------------------------

def ulam_matrix(
    alpha: float,
    n_bins: int = 80,
    n_traj: int = 80_000,
    seed: int = 42,
    beta: float = BETA_PM,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the row-stochastic Ulam transition matrix for the PM map.

    Args:
        alpha: Nonlinearity parameter.
        n_bins: Number of uniform bins in [0,1].
        n_traj: Number of trajectory points.
        seed: Random seed.
        beta: Exponent for PM left branch.

    Returns:
        (P, diagnostics) where P is (n_bins × n_bins) row-stochastic.
    """
    rng = np.random.RandomState(seed)
    T = np.zeros((n_bins, n_bins), dtype=np.float64)

    x = rng.rand(n_traj)
    x_new = pm_map(x, alpha, beta)

    i_bin = np.floor(x * n_bins).astype(np.int32).clip(0, n_bins - 1)
    j_bin = np.floor(x_new * n_bins).astype(np.int32).clip(0, n_bins - 1)
    np.add.at(T, (i_bin, j_bin), 1)

    row_sums = T.sum(axis=1)
    zero_rows = row_sums == 0
    n_zero = int(zero_rows.sum())
    row_sums_safe = row_sums.copy()
    row_sums_safe[zero_rows] = 1.0
    P = T / row_sums_safe[:, None]
    if n_zero > 0:
        P[zero_rows] = 1.0 / n_bins

    actual = P.sum(axis=1)
    row_stoch_error = float(np.max(np.abs(actual - 1.0)))

    diag: Dict[str, Any] = {
        "alpha": alpha,
        "n_bins": n_bins,
        "n_traj": n_traj,
        "seed": seed,
        "n_zero_rows": n_zero,
        "row_stochastic_error": row_stoch_error,
        "min_row_sum": float(actual.min()),
        "max_row_sum": float(actual.max()),
    }
    return P, diag


# ---------------------------------------------------------------------------
# Canonical spectral gap
# ---------------------------------------------------------------------------

def canonical_gap(
    P: np.ndarray,
    diag: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute spectral gap Δ = 1 - |λ₂| with fail-closed guards.

    Returns (gap, augmented_diag). gap is None if any fail condition is met.
    """
    try:
        import numpy.linalg as nla
        ev = nla.eigvals(P)
    except Exception as exc:
        diag["FAIL"] = f"eigvals failed: {exc}"
        return None, diag

    ev_mod = np.abs(ev)
    idx1 = int(np.argmax(ev_mod))
    lambda1_mod = float(ev_mod[idx1])
    lambda1_error = abs(lambda1_mod - 1.0)

    mask = np.ones(len(ev), dtype=bool)
    mask[idx1] = False
    second_modulus = float(np.max(ev_mod[mask]))
    gap = float(1.0 - second_modulus)

    diag.update({
        "lambda1_mod": lambda1_mod,
        "lambda1_error": lambda1_error,
        "second_modulus": second_modulus,
        "gap": gap,
        "eig_max": float(ev_mod.max()),
    })

    if diag["row_stochastic_error"] > ROW_STOCH_TOL:
        diag["FAIL"] = f"row_stochastic_error={diag['row_stochastic_error']:.2e} > {ROW_STOCH_TOL:.0e}"
        return None, diag
    if lambda1_error > LAMBDA1_TOL:
        diag["FAIL"] = f"lambda1_error={lambda1_error:.2e} > {LAMBDA1_TOL:.0e}"
        return None, diag
    if not math.isfinite(gap):
        diag["FAIL"] = f"gap is not finite: {gap}"
        return None, diag

    diag["FAIL"] = None
    return gap, diag


def compute_gap_point(
    alpha: float,
    n_bins: int = 80,
    n_traj: int = 80_000,
    seed: int = 42,
    beta: float = BETA_PM,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute single (alpha → gap) with diagnostics."""
    P, diag = ulam_matrix(alpha, n_bins=n_bins, n_traj=n_traj, seed=seed, beta=beta)
    return canonical_gap(P, diag)


# ---------------------------------------------------------------------------
# Log-log regression
# ---------------------------------------------------------------------------

def loglog_regression(
    alphas: Sequence[float],
    gaps: Sequence[Optional[float]],
    min_gap: float = MIN_GAP,
    n_valid_min: int = N_VALID_MIN,
) -> Dict[str, Any]:
    """Fit log Δ = c + κ·log α over valid gap points (gap > min_gap).

    Returns dict with kappa_fit, R2, RMS_resid, stderr_kappa, CI95_kappa,
    n_valid, qualificatif. FAIL key is set to a string on failure, else None.
    """
    gaps_arr = np.array([g if g is not None else 0.0 for g in gaps])
    alphas_arr = np.array(alphas, dtype=float)
    valid = np.array([(g is not None and g > min_gap) for g in gaps])
    n_valid = int(valid.sum())
    result: Dict[str, Any] = {"n_valid": n_valid, "n_total": len(alphas)}

    if n_valid < n_valid_min:
        result["FAIL"] = f"n_valid={n_valid} < {n_valid_min}"
        result["kappa_fit"] = None
        return result

    log_a = np.log(alphas_arr[valid])
    log_g = np.log(gaps_arr[valid])

    A = np.column_stack([np.ones(n_valid), log_a])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_g, rcond=None)
    except Exception as exc:
        result["FAIL"] = f"lstsq failed: {exc}"
        result["kappa_fit"] = None
        return result

    c_fit, kappa_fit = float(coeffs[0]), float(coeffs[1])
    log_g_pred = c_fit + kappa_fit * log_a
    residuals = log_g - log_g_pred

    SS_res = float(np.sum(residuals ** 2))
    SS_tot = float(np.sum((log_g - float(log_g.mean())) ** 2))
    R2 = float(1.0 - SS_res / SS_tot) if SS_tot > 0 else 0.0
    RMS = float(np.sqrt(SS_res / n_valid))

    sigma2 = SS_res / max(n_valid - 2, 1)
    XtX_inv = np.linalg.pinv(A.T @ A)
    stderr_k = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t_crit = 2.0
    CI95 = (float(kappa_fit - t_crit * stderr_k), float(kappa_fit + t_crit * stderr_k))

    if R2 >= R2_PUB and RMS <= RMS_GOOD and stderr_k <= STDERR_PUB:
        qual = "PUBLICATION_GRADE"
    elif R2 >= R2_GOOD and RMS <= RMS_GOOD:
        qual = "GOOD_SCALING"
    else:
        qual = "PRE_ASYMPTOTIC"

    D = abs(kappa_fit - KAPPA_THEORY)
    result.update({
        "FAIL": None,
        "kappa_fit": round(kappa_fit, 6),
        "c_fit": round(c_fit, 6),
        "R2": round(R2, 6),
        "RMS_resid": round(RMS, 6),
        "stderr_kappa": round(stderr_k, 6),
        "CI95_kappa": [round(CI95[0], 4), round(CI95[1], 4)],
        "D": round(D, 6),
        "kappa_theory": KAPPA_THEORY,
        "qualificatif": qual,
    })
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    alpha_grid: Sequence[float],
    n_bins: int = 80,
    n_traj: int = 80_000,
    seed: int = 42,
    beta: float = BETA_PM,
    min_gap: float = MIN_GAP,
) -> Dict[str, Any]:
    """Run the full Ulam spectral gap pipeline over an alpha grid.

    Computes gap for each alpha, fits log-log regression, and returns a summary dict.

    Args:
        alpha_grid: Sequence of α values (e.g. np.geomspace(0.01, 0.20, 12)).
        n_bins: Ulam bin count.
        n_traj: Trajectory count per matrix construction.
        seed: RNG seed.
        beta: PM map exponent.
        min_gap: Minimum gap to include in regression.

    Returns dict with alphas, gaps, regression results, and fail-closed status.
    """
    alpha_list = list(alpha_grid)
    gaps: List[Optional[float]] = []
    diags: List[Dict[str, Any]] = []
    any_fail = False
    row_stoch_max = 0.0
    lambda1_err_max = 0.0

    for alpha in alpha_list:
        gap, diag = compute_gap_point(alpha, n_bins=n_bins, n_traj=n_traj, seed=seed, beta=beta)
        gaps.append(gap)
        diags.append(diag)
        if diag.get("FAIL"):
            any_fail = True
        else:
            row_stoch_max = max(row_stoch_max, diag["row_stochastic_error"])
            lambda1_err_max = max(lambda1_err_max, diag["lambda1_error"])

    reg = loglog_regression(alpha_list, gaps, min_gap=min_gap)

    return {
        "beta": beta,
        "kappa_theory": KAPPA_THEORY,
        "n_bins": n_bins,
        "n_traj": n_traj,
        "seed": seed,
        "alphas": alpha_list,
        "gaps": gaps,
        "row_stochastic_error_max": round(row_stoch_max, 14),
        "lambda1_error_max": round(lambda1_err_max, 14),
        "any_fail_point": any_fail,
        "regression": reg,
        "kappa_fit": reg.get("kappa_fit"),
        "R2": reg.get("R2"),
        "D": reg.get("D"),
        "qualificatif": reg.get("qualificatif"),
        "regression_fail": reg.get("FAIL"),
        "claim_ceiling": "LOCAL_LAB_ONLY_NOT_EXTERNAL_PROOF",
    }
