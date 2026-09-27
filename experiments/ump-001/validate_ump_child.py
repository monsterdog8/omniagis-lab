#!/usr/bin/env python3
"""Fail-closed structural validator for UMP-001 patch-forward child artifacts."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_PARENT = "3d000c5ec1ec27d1fde68fa528ac25e9d5c34b698e3cc7f684682cf180f3057c"


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def require(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    binding = load("GTAB_BIND_001.json")
    pilot = load("UMP_001_RC3_ENGINEERING_PILOT_CONFIG_V1.json")
    sync = load("UMP_001_HANGAR_SYNC_001.json")

    require(binding["parent"]["stage0_capsule_sha256"] == EXPECTED_PARENT, "binding parent hash drift")
    require(binding["parent"]["immutable"] is True, "binding must preserve immutable Stage0 parent")
    require(binding["status"] == "PROPOSED_NOT_EXECUTED", "binding status overpromoted")
    require(binding["claim_ceiling"] == "BINDING_SPEC_ONLY", "binding claim ceiling drift")
    require(binding["gates"]["EMPIRICAL_C2_EVIDENCE"] == 0, "C2 evidence must remain zero before real trials")
    require(binding["gates"]["EMPIRICAL_C4_EVIDENCE"] == 0, "C4 evidence must remain zero before real trials")
    require(binding["gates"]["OMEGA_U"] == "NOT_MEASURED", "Omega_U must remain NOT_MEASURED")
    require(binding["gates"]["POSTCOMMIT_PHYSICAL_INDEPENDENCE"] == "NOT_PROVEN",
            "physical RNG independence may not be promoted by specification")
    require(binding["confirmatory_design"]["D_PRIMARY"] == "TBD", "D* must remain TBD")
    require(binding["confirmatory_design"]["DELTA_PRIMARY"] == "TBD", "Delta* must remain TBD")

    require(pilot["parent_stage0_capsule_sha256"] == EXPECTED_PARENT, "pilot parent hash drift")
    require(pilot["status"] == "BLOCKED_NOT_AUTHORIZED", "pilot must remain blocked")
    require(pilot["readiness_rule"]["pilot_authorized"] is False, "pilot must not be authorized")
    require(pilot["pilot_design"]["confirmatory_reuse_forbidden"] is True,
            "engineering pilot data reuse into confirmatory must be forbidden")

    require(sync["parent_stage0"]["capsule_sha256"] == EXPECTED_PARENT, "hangar sync parent hash drift")
    require(sync["methodological_sync"]["universal_connection"] == "NOT_SUPPORTED",
            "universal connection must remain NOT_SUPPORTED")
    require(sync["current_gates"]["EMPIRICAL_C2_EVIDENCE"] == 0, "sync C2 evidence must be zero")
    require(sync["current_gates"]["EMPIRICAL_C4_EVIDENCE"] == 0, "sync C4 evidence must be zero")
    require(sync["current_gates"]["OMEGA_U"] == "NOT_MEASURED", "sync Omega_U must be NOT_MEASURED")

    print("UMP_001_CHILD_STRUCTURAL_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"UMP_001_CHILD_STRUCTURAL_VALIDATION_FAIL: {exc}", file=sys.stderr)
        raise
