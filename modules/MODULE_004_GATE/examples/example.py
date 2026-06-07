"""Minimal usage examples for MODULE_004_GATE.

Run from repository root:
    python modules/MODULE_004_GATE/examples/example.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from gate import (
    classify_claim, ClaimStatus,
    from_gate_counts, compute_evidence_score,
    maturity_map, proof_firewall,
    EVIDENCE_WEIGHTS, MATURITY_STAGES,
)


# ---------------------------------------------------------------------------
# Example 1: Claim classification
# ---------------------------------------------------------------------------

def example_claim_classification():
    print("=== Example 1: classify_claim() ===")

    claims = [
        "This is a hypothesis, might be testable as LAB_ONLY prototype",
        "Our system is SOTA and beats all benchmarks",
        "This is production-ready and approved for production",
        "We tested structure analysis over 30 minutes of raw data",
        "consciousness proven in digital system",
        "The output quality is generally acceptable",
    ]

    for text in claims:
        result = classify_claim(text)
        short = text[:55] + "..." if len(text) > 55 else text
        print(f"  [{result.status:<24}] {short!r}")
        if result.hits:
            print(f"    hits: {result.hits}")

    print()


# ---------------------------------------------------------------------------
# Example 2: Evidence scoring from gate counts
# ---------------------------------------------------------------------------

def example_evidence_scoring():
    print("=== Example 2: Evidence scoring from gate counts ===")

    scenarios = [
        ("No data collected",
         dict(expected=0, present=0, valid=0)),
        ("Partial RAW collection (3/10 valid)",
         dict(expected=10, present=10, valid=3, safety=0.7)),
        ("Full RAW, no scoring/replay",
         dict(expected=10, present=10, valid=10, safety=0.8)),
        ("Full RAW + scoring",
         dict(expected=10, present=10, valid=10, safety=0.9, scoring=True)),
        ("Full evidence chain",
         dict(expected=10, present=10, valid=10, safety=1.0,
              scoring=True, replay=True, independence=True)),
    ]

    print(f"  {'Scenario':<38} {'final_score':<14} {'status':<25} {'public_ok'}")
    print("  " + "-" * 90)
    for label, kwargs in scenarios:
        inp = from_gate_counts(**kwargs)
        score = compute_evidence_score(inp)
        print(f"  {label:<38} {score.final_score:<14.4f} {score.scoring_status:<25} {score.public_claim_allowed}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Maturity map and proof firewall
# ---------------------------------------------------------------------------

def example_maturity_and_firewall():
    print("=== Example 3: Maturity map and proof firewall ===")

    # Maturity at RAW stage
    mat = maturity_map(idea=True, design=True, prototype=True, raw=True)
    print(f"  highest_maturity   : {mat['highest_maturity']}")
    print(f"  public_claim_right : {mat['public_claim_right']}")
    print(f"  production_status  : {mat['production_status']}")
    print()

    # Proof firewall — one dimension missing
    fw_blocked = proof_firewall({
        "DATA": 0.0, "RAW": 0.8, "SCORING": 0.8,
        "REPLAY": 0.8, "INDEPENDENCE": 0.8, "SAFETY": 0.9,
    })
    print(f"  Firewall (DATA=0): {fw_blocked['verdict']}")
    print(f"    missing: {fw_blocked['missing']}")

    # Proof firewall — all dimensions present
    fw_ready = proof_firewall({
        "DATA": 0.9, "RAW": 0.9, "SCORING": 0.9,
        "REPLAY": 0.9, "INDEPENDENCE": 0.9, "SAFETY": 0.9,
    })
    print(f"  Firewall (all ok): {fw_ready['verdict']}")
    print()

    print(f"  EVIDENCE_WEIGHTS: {EVIDENCE_WEIGHTS}")
    print(f"  MATURITY_STAGES:  {MATURITY_STAGES}")
    print()


if __name__ == "__main__":
    example_claim_classification()
    example_evidence_scoring()
    example_maturity_and_firewall()
