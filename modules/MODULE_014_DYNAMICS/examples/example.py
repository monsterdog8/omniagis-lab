"""Minimal usage examples for MODULE_014_DYNAMICS.

Run from repository root:
    python modules/MODULE_014_DYNAMICS/examples/example.py

No external dependencies required (pure stdlib).
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from dynamics import (
    ContinuumState,
    energy,
    step_continuum,
    run_continuum,
    FractalEngine,
    FractalState,
)


# ---------------------------------------------------------------------------
# Example 1: ContinuumState — single step with zero noise
# ---------------------------------------------------------------------------

def example_continuum_single_step():
    print("=== Example 1: ContinuumState single deterministic step ===")
    s0 = ContinuumState(psi=0.95, F=0.88, S=0.02, C=0.10, R=55.435, Q=0.0)
    s0.Q = energy(s0)
    print(f"  Initial: psi={s0.psi:.4f}  F={s0.F:.4f}  S={s0.S:.5f}  C={s0.C:.4f}  R={s0.R:.3f}  Q={s0.Q:.4f}")

    s1 = step_continuum(s0, noise_std=0.0)
    print(f"  After 1 step (no noise): psi={s1.psi:.4f}  F={s1.F:.4f}  S={s1.S:.5f}  C={s1.C:.4f}  Q={s1.Q:.4f}")
    print()


# ---------------------------------------------------------------------------
# Example 2: run_continuum — short trajectory with fixed seed
# ---------------------------------------------------------------------------

def example_run_continuum():
    print("=== Example 2: run_continuum — 20-step trajectory (seed=42) ===")
    history = run_continuum(n_steps=20, seed=42, noise_std=0.002)
    print(f"  States returned: {len(history)} (n_steps+1)")
    first = history[0]
    last  = history[-1]
    print(f"  Step  0: psi={first['psi']:.4f}  Q={first['Q']:.4f}")
    print(f"  Step 20: psi={last['psi']:.4f}  Q={last['Q']:.4f}")
    psi_vals = [h["psi"] for h in history]
    print(f"  psi range: [{min(psi_vals):.4f}, {max(psi_vals):.4f}]")
    print()


# ---------------------------------------------------------------------------
# Example 3: FractalEngine — coherence tracking over 30 cycles
# ---------------------------------------------------------------------------

def example_fractal_engine():
    print("=== Example 3: FractalEngine — 30 cycles of coherence tracking ===")
    engine = FractalEngine(history_size=100)

    for i in range(30):
        state = engine.compute_metrics(cycle_id=i)

    stats = engine.get_statistics()
    print(f"  Cycles computed : {engine.cycle_count}")
    print(f"  coherence (last): {state.coherence:.7f}")
    print(f"  entropy   (last): {state.entropy:.7f}")
    print(f"  resonance_hz    : {state.resonance_hz:.6f}")
    print(f"  drift           : {state.drift:.8f}")
    print(f"  History stats   : mean={stats['mean']:.7f}  std={stats['std']:.8f}")
    print(f"  Is stable?      : {engine.is_stable()}")
    print()


if __name__ == "__main__":
    example_continuum_single_step()
    example_run_continuum()
    example_fractal_engine()
