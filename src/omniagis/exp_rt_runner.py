"""
exp_rt_runner.py — experiment runner for return-time validation.

Uses the Pomeau-Manneville (type-I intermittency) map as the canonical
intermittent dynamical system:

    f(x) = (x + x^z) mod 1,    z > 1

The target set is  U = [0, epsilon].  Return times  tau_i = t_{i+1} - t_i
are the primary observable.

For z > 1 the invariant measure near x = 0 is finite (z < 2) or infinite
(z >= 2), and the return-time survival function obeys

    S(n) = P(tau > n) ~ n^{-alpha},    alpha = 1 / (z - 1)

so the theoretical exponent is  alpha_theory = 1 / (z - 1).

Usage
-----
As a module::

    from omniagis.exp_rt_runner import run_experiment, ExperimentParams
    result = run_experiment(ExperimentParams(z=1.5, epsilon=0.1))

As a CLI script::

    python -m omniagis.exp_rt_runner --z 1.5 --epsilon 0.1 --output result.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .validatorgate_full import ValidationConfig, validate


# ---------------------------------------------------------------------------
# Pomeau-Manneville map
# ---------------------------------------------------------------------------

def pomeau_manneville_step(x: float, z: float) -> float:
    """One iterate of the Pomeau-Manneville map:  f(x) = (x + x^z) mod 1."""
    return (x + x ** z) % 1.0


def generate_trajectory(
    z: float,
    n_steps: int,
    x0: float = 0.5,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Generate a trajectory of length ``n_steps`` from the PM map.

    Parameters
    ----------
    z :
        Intermittency exponent (z > 1).  Theory predicts
        alpha = 1 / (z - 1) for the return-time power-law exponent.
    n_steps :
        Number of iterates to generate (trajectory length).
    x0 :
        Initial condition x(0).
    rng :
        Unused (the PM map is fully deterministic given z and x0).
        Accepted so that callers can pass a generator without branching.
    """
    traj = np.empty(n_steps, dtype=np.float64)
    x = float(x0)
    for i in range(n_steps):
        traj[i] = x
        x = pomeau_manneville_step(x, z)
    return traj


# ---------------------------------------------------------------------------
# Experiment parameters & runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentParams:
    z: float = 1.5
    """Intermittency exponent (z > 1)."""

    epsilon: float = 0.1
    """Target-set radius: U = [0, epsilon]."""

    n_steps: int = 1_000_000
    """Trajectory length.  Longer trajectories improve tail statistics."""

    x0: float = 0.5
    """Initial condition."""

    n_min: int = 5
    """Lower bound for power-law tail fitting (n >= n_min)."""

    min_tail_sample_size: int = 1000
    """Minimum count(tau >= n_min) required; fit rejected below this."""

    min_tail_obs: int = 10
    """Minimum observations backing each S(n) regression point."""

    min_r_squared: float = 0.80
    """Minimum R² for the power-law fit."""

    require_plateau: bool = False
    """If True, ACCEPTED requires a detected scale-invariant plateau."""

    n_bootstrap: int = 500
    """Bootstrap resamples for CI estimation."""

    seed: int = 42
    """Random seed for full reproducibility."""


def run_experiment(params: ExperimentParams) -> dict:
    """Run the full validation pipeline for one parameter set.

    Returns a JSON-serialisable dict containing experiment metadata and
    the complete structured validation report (all six checks).
    """
    if params.z <= 1.0:
        raise ValueError(f"z must be > 1.0, got {params.z}")

    rng = np.random.default_rng(params.seed)
    traj = generate_trajectory(params.z, params.n_steps, params.x0, rng=rng)

    theory_alpha = 1.0 / (params.z - 1.0)
    in_target = lambda x: (x >= 0.0) & (x <= params.epsilon)

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

    report = validate(traj, in_target, config=config, rng=rng)

    return {
        "experiment": asdict(params),
        "theory_alpha": theory_alpha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run return-time validation for the Pomeau-Manneville map "
            "and write a JSON report."
        )
    )
    p.add_argument("--z", type=float, default=1.5,
                   help="Intermittency exponent (default: 1.5)")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Target-set radius [0, ε] (default: 0.1)")
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
                   help="Minimum R² for power-law fit (default: 0.80)")
    p.add_argument("--require-plateau", action="store_true",
                   help="Require plateau detection for ACCEPTED verdict")
    p.add_argument("--n-bootstrap", type=int, default=500,
                   help="Bootstrap resamples (default: 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output", default=None,
                   help="Output JSON file (default: stdout)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    params = ExperimentParams(
        z=args.z,
        epsilon=args.epsilon,
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
    result = run_experiment(params)
    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
