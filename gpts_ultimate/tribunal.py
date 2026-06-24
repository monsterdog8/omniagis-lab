"""Unified verdict engine — gpts_ultimate.

Combines BERZERKER (logistic lattice ablation) and WorldAgnostic (any time series)
tribunal systems into a single API. Uses the more conservative verdict when both
are run.

Verdicts: REFUTED_VS_NULL | OPEN_PROBLEM | EMPIRICAL_SIGNAL | PASS_LOCAL
Claim ceiling: LOCAL_SIMULATION_ONLY
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from gpts_ultimate import berzerker as _bz
from gpts_ultimate import agnostic as _ag

_VERDICT_RANK = {
    "REFUTED_VS_NULL": 0,
    "OPEN_PROBLEM": 1,
    "EMPIRICAL_SIGNAL": 2,
    "PASS_LOCAL": 3,
}


def _conservative(v1: str, v2: str) -> str:
    """Return the more conservative (lower-ranked) of two verdict strings."""
    return v1 if _VERDICT_RANK.get(v1, 0) <= _VERDICT_RANK.get(v2, 0) else v2


def run_berzerker_tribunal(
    C: np.ndarray,
    seeds: Sequence[int],
    thr: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Run BERZERKER ablation tribunal on a coupling matrix C.

    Delegates to berzerker.tribunal_verdict.
    Returns full verdict dict with claim_ceiling=LOCAL_SIMULATION_ONLY.
    """
    return _bz.tribunal_verdict(C, seeds, thr=thr, output_path=output_path)


def run_agnostic_tribunal(
    metrics_real: Dict[str, float],
    null_dist: Dict,
) -> Dict:
    """Run world-agnostic tribunal on pre-computed metrics vs. null distribution.

    Delegates to agnostic.tribunal_verdict.
    Returns verdict dict with claim_ceiling=LOCAL_SIMULATION_ONLY.
    """
    return _ag.tribunal_verdict(metrics_real, null_dist)


def _build_null_dist(trajectory: np.ndarray, n_null: int = 40, base_seed: int = 0) -> Dict:
    """Build null distribution for world-agnostic metrics via phase scrambling."""
    from gpts_ultimate.agnostic import WorldAgnosticNulls, WorldAgnosticMetrics, compute_metrics

    pool: Dict[str, list] = {}
    for j in range(n_null):
        null_s = WorldAgnosticNulls.phase_scramble(trajectory, seed=base_seed + j)
        m = compute_metrics(null_s)
        for k, v in m.items():
            pool.setdefault(k, []).append(v)

    return {
        k: {
            "p95": float(np.percentile(vals, 95)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        for k, vals in pool.items()
    }


def run_full_tribunal(
    trajectory: np.ndarray,
    lorenz_ensemble: Optional[np.ndarray] = None,
    seeds: Sequence[int] = range(5, 25),
    n_null: int = 40,
    output_path: Optional[str] = None,
) -> Dict:
    """Full tribunal combining world-agnostic + optional BERZERKER verdicts.

    Steps:
      1. Compute WorldAgnosticMetrics on trajectory.
      2. Build empirical null distribution via n_null phase-scrambled surrogates.
      3. Run agnostic tribunal → verdict_agnostic.
      4. If lorenz_ensemble provided: run BERZERKER on default coupling → verdict_berzerker.
      5. Merge: final_verdict = more conservative of the two.

    Returns consolidated dict with keys:
      agnostic, berzerker (if run), final_verdict, claim_ceiling, production_unlocked.
    """
    from gpts_ultimate.agnostic import compute_metrics

    metrics_real = compute_metrics(trajectory, all_channels=lorenz_ensemble)
    null_dist = _build_null_dist(trajectory, n_null=n_null)
    verdict_agnostic = run_agnostic_tribunal(metrics_real, null_dist)

    result: Dict = {
        "agnostic": verdict_agnostic,
        "berzerker": None,
        "final_verdict": verdict_agnostic["verdict"],
        "claim_ceiling": "LOCAL_SIMULATION_ONLY",
        "production_unlocked": False,
    }

    if lorenz_ensemble is not None:
        C = _bz.base_coupling()
        verdict_berzerker = run_berzerker_tribunal(C, list(seeds), output_path=output_path)
        result["berzerker"] = verdict_berzerker
        result["final_verdict"] = _conservative(
            verdict_agnostic["verdict"],
            verdict_berzerker["verdict"],
        )

    return result
