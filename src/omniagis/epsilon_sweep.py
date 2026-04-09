"""
epsilon_sweep.py — ε-sweep robustness analysis for return-time validation.

Runs the full validation pipeline across a log-spaced range of ε values
and produces a structured JSON report summarising how each check and the
power-law exponent vary with the target-set radius ε.

A **single trajectory** is generated once and shared across all ε values so
that differences in outcomes are attributable to ε alone (not sampling
variation).

The sweep tests:
- Whether the verdict is stable (robust) across ε values.
- Whether the fitted alpha is consistent with the theoretical prediction.
- How CI width changes with ε (more returns → narrower CI for larger ε).

Usage
-----
As a module::

    from omniagis.epsilon_sweep import run_epsilon_sweep, SweepParams
    report = run_epsilon_sweep(SweepParams(z=1.5))

As a CLI script::

    python -m omniagis.epsilon_sweep --z 1.5 --output sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .exp_rt_runner import generate_trajectory
from .validatorgate_full import ValidationConfig, validate


# ---------------------------------------------------------------------------
# Per-ε helper
# ---------------------------------------------------------------------------

def _validate_for_epsilon(
    trajectory: np.ndarray,
    epsilon: float,
    config: ValidationConfig,
) -> dict:
    """Run validation for one ε value; return a structured result row."""
    in_target = lambda x: (x >= 0.0) & (x <= epsilon)
    report = validate(trajectory, in_target, config=config)

    pl = report["diagnostics"]["power_law"]
    ms = report["diagnostics"]["multi_scale_ci"]

    return {
        "epsilon": epsilon,
        "verdict": report["verdict"],
        "n_return_times": report["n_return_times"],
        "checks": report["checks"],
        "fail_reasons": report["fail_reasons"],
        "power_law": {
            "alpha": pl["alpha"],
            "r_squared": pl["r_squared"],
            "tail_sample_size": pl["tail_sample_size"],
            "n_tail_points": pl["n_tail_points"],
        },
        "multi_scale_ci": {
            "scales": ms["scales"],
            "width": ms["width"],
            "alpha_hat": ms["alpha_hat"],
            "lower": ms["lower"],
            "upper": ms["upper"],
            "ci_shrinks": ms["ci_shrinks"],
        },
        "tail_mass": report["diagnostics"]["tail_mass"],
        "plateau_detected": report["diagnostics"]["plateau"]["detected"],
    }


# ---------------------------------------------------------------------------
# Sweep parameters & runner
# ---------------------------------------------------------------------------

@dataclass
class SweepParams:
    z: float = 1.5
    """Intermittency exponent (z > 1).  theory_alpha = 1 / (z - 1)."""

    epsilon_min: float = 0.02
    """Smallest ε in the sweep (log-space lower bound)."""

    epsilon_max: float = 0.20
    """Largest ε in the sweep (log-space upper bound)."""

    n_epsilons: int = 10
    """Number of ε values (log-spaced)."""

    n_steps: int = 1_000_000
    """Trajectory length shared across all ε values."""

    x0: float = 0.5
    """Initial condition."""

    n_min: int = 5
    """Lower bound for power-law tail fitting."""

    min_tail_sample_size: int = 1000
    """Minimum tail observations required for a valid fit."""

    min_tail_obs: int = 10
    """Minimum observations backing each S(n) regression point."""

    min_r_squared: float = 0.80
    """Minimum R² for the power-law fit."""

    require_plateau: bool = False
    """If True, ACCEPTED requires a detected scale-invariant plateau."""

    n_bootstrap: int = 200
    """Bootstrap resamples per ε (reduced for sweep speed)."""

    seed: int = 42
    """Random seed for full reproducibility."""


def run_epsilon_sweep(params: SweepParams) -> dict:
    """Run validation for a log-spaced sweep of ε values.

    The trajectory is generated once with the given seed; all ε values
    share the same underlying orbit so that results are comparable.

    Returns
    -------
    JSON-serialisable dict with:

    ``sweep_params``
        The parameters used for this sweep.
    ``theory_alpha``
        Theoretical exponent  1 / (z - 1).
    ``timestamp``
        UTC ISO-8601 timestamp.
    ``summary``
        Aggregate statistics over all ε values.
    ``results``
        Per-ε structured result rows (one per ε value).
    """
    rng = np.random.default_rng(params.seed)
    traj = generate_trajectory(params.z, params.n_steps, params.x0, rng=rng)
    theory_alpha = 1.0 / (params.z - 1.0)

    epsilons = np.logspace(
        math.log10(params.epsilon_min),
        math.log10(params.epsilon_max),
        params.n_epsilons,
    )

    config = ValidationConfig(
        n_min=params.n_min,
        min_tail_sample_size=params.min_tail_sample_size,
        min_tail_obs=params.min_tail_obs,
        min_r_squared=params.min_r_squared,
        theory_alpha=theory_alpha,
        require_plateau=params.require_plateau,
        n_bootstrap=params.n_bootstrap,
        seed=params.seed,
    )

    results = [
        _validate_for_epsilon(traj, float(eps), config)
        for eps in epsilons
    ]

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    n_accepted = sum(1 for r in results if r["verdict"] == "ACCEPTED")
    n_fail = len(results) - n_accepted
    acceptance_rate = n_accepted / len(results) if results else float("nan")

    # Collect valid (non-NaN) alpha estimates from best scale (index 0)
    alphas_best = [
        r["multi_scale_ci"]["alpha_hat"][0]
        for r in results
        if not math.isnan(r["multi_scale_ci"]["alpha_hat"][0])
    ]

    # CI widths at best scale
    widths_best = [
        r["multi_scale_ci"]["width"][0]
        for r in results
        if not math.isnan(r["multi_scale_ci"]["width"][0])
    ]

    def _safe_stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(vals),
        }

    # Per-check acceptance counts
    check_names = ["R2_ok", "CI_width_ok", "theory_in_CI",
                   "CI_shrinks", "plateau", "tail_mass"]
    check_pass_counts = {
        name: sum(1 for r in results if r["checks"].get(name, False))
        for name in check_names
    }

    summary = {
        "n_epsilons": len(results),
        "n_accepted": n_accepted,
        "n_fail_closed": n_fail,
        "acceptance_rate": acceptance_rate,
        "theory_alpha": theory_alpha,
        "alpha_best_scale": _safe_stats(alphas_best),
        "ci_width_best_scale": _safe_stats(widths_best),
        "check_pass_counts": check_pass_counts,
    }

    return {
        "sweep_params": asdict(params),
        "theory_alpha": theory_alpha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ε-sweep robustness analysis for return-time validation."
    )
    p.add_argument("--z", type=float, default=1.5,
                   help="Intermittency exponent (default: 1.5)")
    p.add_argument("--epsilon-min", type=float, default=0.02,
                   help="Minimum ε (default: 0.02)")
    p.add_argument("--epsilon-max", type=float, default=0.20,
                   help="Maximum ε (default: 0.20)")
    p.add_argument("--n-epsilons", type=int, default=10,
                   help="Number of ε values (default: 10)")
    p.add_argument("--n-steps", type=int, default=1_000_000,
                   help="Trajectory length (default: 1 000 000)")
    p.add_argument("--x0", type=float, default=0.5,
                   help="Initial condition (default: 0.5)")
    p.add_argument("--n-min", type=int, default=5,
                   help="Min n for tail fitting (default: 5)")
    p.add_argument("--min-tail-sample-size", type=int, default=1000,
                   help="Minimum tail observations (default: 1000)")
    p.add_argument("--min-tail-obs", type=int, default=10,
                   help="Min observations per regression point (default: 10)")
    p.add_argument("--min-r-squared", type=float, default=0.80,
                   help="Minimum R² (default: 0.80)")
    p.add_argument("--require-plateau", action="store_true",
                   help="Require plateau detection for ACCEPTED")
    p.add_argument("--n-bootstrap", type=int, default=200,
                   help="Bootstrap resamples per ε (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output", default=None,
                   help="Output JSON file (default: stdout)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    params = SweepParams(
        z=args.z,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        n_epsilons=args.n_epsilons,
        n_steps=args.n_steps,
        x0=args.x0,
        n_min=args.n_min,
        min_tail_sample_size=args.min_tail_sample_size,
        min_tail_obs=args.min_tail_obs,
        min_r_squared=args.min_r_squared,
        require_plateau=args.require_plateau,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    result = run_epsilon_sweep(params)
    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(output_json)
        print(f"Sweep report written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
