#!/usr/bin/env python3
"""UMP-001 RC3 confirmatory power planner.

This is a patch-forward planning tool. It refuses to produce a canonical
confirmatory N unless pilot-derived primary-score variance and a preregistered
practical effect threshold epsilon are supplied.

Primary unit: independent session/trial block.
Primary endpoint: paired proper-score skill versus the exact randomized null.
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist

PARENT_STAGE0_SHA256 = "3d000c5ec1ec27d1fde68fa528ac25e9d5c34b698e3cc7f684682cf180f3057c"


def normal_approx_n(*, sigma: float, epsilon: float, alpha: float, power: float) -> int:
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be finite and >0")
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and >0")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0,1)")
    z_alpha = NormalDist().inv_cdf(1 - alpha)
    z_power = NormalDist().inv_cdf(power)
    return math.ceil(((z_alpha + z_power) * sigma / epsilon) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-sd", type=float, default=None,
                    help="Pilot SD of independent-block primary score deltas.")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Preregistered minimum practically meaningful positive score skill.")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--power", type=float, default=0.90)
    args = ap.parse_args()

    report = {
        "schema": "UMP_001_RC3_POWER_PLAN_RUNTIME_V1",
        "parent_stage0_capsule_sha256": PARENT_STAGE0_SHA256,
        "primary_endpoint": "PAIRED_PROPER_SCORE_SKILL_VS_EXACT_RANDOMIZED_NULL",
        "statistical_unit": "INDEPENDENT_TRIAL_OR_SESSION_BLOCK",
        "alpha_one_sided": args.alpha,
        "target_power": args.power,
        "pilot_sd": args.pilot_sd,
        "epsilon": args.epsilon,
        "method": "NORMAL_APPROXIMATION_PLANNING_ONLY",
        "canonical_confirmatory_N": "NOT_COMPUTED",
        "status": "BLOCKED_MISSING_PILOT_SD_AND_EPSILON",
        "claim_ceiling": "POWER_PLANNING_ONLY"
    }

    if args.pilot_sd is not None and args.epsilon is not None:
        n = normal_approx_n(
            sigma=args.pilot_sd,
            epsilon=args.epsilon,
            alpha=args.alpha,
            power=args.power,
        )
        report["engineering_N_estimate"] = n
        report["status"] = "ENGINEERING_ESTIMATE_AVAILABLE_REQUIRES_PREREG_CONFIRMATION"
        report["canonical_confirmatory_N"] = "NOT_FROZEN"

    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
