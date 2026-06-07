"""Minimal usage examples for MODULE_016_CORE_INIT.

Demonstrates importing all major symbol groups directly from gpts_core
without referencing sub-modules.

Run from repository root:
    python modules/MODULE_016_CORE_INIT/examples/example.py
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Example 1: Gate — classify a claim and compute evidence score
# ---------------------------------------------------------------------------

def example_gate():
    print("=== Example 1: gate symbols via gpts_core top-level ===")

    from gpts_core import classify_claim, from_gate_counts, compute_evidence_score

    result = classify_claim("This is a local lab-only hypothesis not approved for production")
    print(f"  classify_claim status : {result.status}")

    inp = from_gate_counts(expected=10, present=10, valid=10,
                           safety=1.0, scoring=True, replay=True, independence=True)
    score = compute_evidence_score(inp)
    print(f"  evidence final_score  : {score.final_score:.3f}")
    print(f"  public_claim_allowed  : {score.public_claim_allowed}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Dynamics — ContinuumState and FractalEngine
# ---------------------------------------------------------------------------

def example_dynamics():
    print("=== Example 2: dynamics symbols via gpts_core top-level ===")

    from gpts_core import ContinuumState, step_continuum, continuum_energy, FractalEngine

    s0 = ContinuumState()
    s0.Q = continuum_energy(s0)
    s1 = step_continuum(s0, noise_std=0.0)
    print(f"  Initial Q  : {s0.Q:.4f}")
    print(f"  After step : psi={s1.psi:.4f}  Q={s1.Q:.4f}")

    engine = FractalEngine(history_size=10)
    for i in range(10):
        engine.compute_metrics(cycle_id=i)
    stats = engine.get_statistics()
    print(f"  Fractal coherence mean : {stats['mean']:.7f}")
    print(f"  Fractal is_stable      : {engine.is_stable()}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Coherence passport — build from top-level imports
# ---------------------------------------------------------------------------

def example_coherence():
    print("=== Example 3: coherence symbols via gpts_core top-level ===")

    from gpts_core import (
        shannon_entropy, global_coherence, build_coherence_passport, seal_passport_hashes
    )

    symbols = [0, 1] * 50
    h = shannon_entropy(symbols)
    print(f"  shannon_entropy([0,1]*50)  : {h:.4f} bits")

    c = global_coherence(0.6, 1.2)
    print(f"  global_coherence(0.6, 1.2) : {c:.4f}")

    obs = {"mod_A": [float(i) for i in range(20)],
           "mod_B": [float(i % 5) for i in range(20)]}
    passport = build_coherence_passport("C_DEMO", obs)
    passport = seal_passport_hashes(passport)
    print(f"  Passport cycle_id          : {passport.get('cycle_id')}")
    print(f"  Passport has hash          : {'hash' in str(passport)}")
    print()


if __name__ == "__main__":
    example_gate()
    example_dynamics()
    example_coherence()
