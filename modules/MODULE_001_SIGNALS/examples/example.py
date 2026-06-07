"""Minimal usage examples for MODULE_001_SIGNALS.

Run from repository root:
    python modules/MODULE_001_SIGNALS/examples/example.py
"""
from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import numpy as np
from signals import analyze, bootstrap_ci, generate_null, step_metrics

# ---------------------------------------------------------------------------
# Example 1: Analyze a sine wave signal
# ---------------------------------------------------------------------------

def example_analyze_sine():
    sr = 256.0
    n = 512
    freq = 10.0
    x = [math.sin(2 * math.pi * freq * i / sr) for i in range(n)]

    metrics = analyze(x, sr=sr)

    print("=== Example 1: analyze() on a 10 Hz sine wave ===")
    print(f"  structure_score      : {metrics['structure_score']:.4f}")
    print(f"  spectral_peak_hz     : {metrics['spectral_peak_hz']:.2f} Hz")
    print(f"  spectral_concentration: {metrics['spectral_concentration']:.4f}")
    print(f"  autocorr_max         : {metrics['autocorr_max']:.4f}")
    print(f"  entropy_norm         : {metrics['entropy_norm']:.4f}")
    print(f"  ar2_r2               : {metrics['ar2_r2']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Null model generation and bootstrap CI
# ---------------------------------------------------------------------------

def example_null_and_bootstrap():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 256)

    null_shuffle = generate_null(x, seed=0, kind="shuffle")
    null_phase   = generate_null(x, seed=0, kind="phase_scramble")

    from signals import analyze
    real_scores  = [analyze(x, sr=256.0)["structure_score"]]
    null_scores  = [analyze(null_shuffle, sr=256.0)["structure_score"],
                    analyze(null_phase,   sr=256.0)["structure_score"]]

    ci = bootstrap_ci(null_scores * 50, seed=7, n_boot=200)

    print("=== Example 2: Null model comparison + bootstrap CI ===")
    print(f"  real structure_score  : {real_scores[0]:.4f}")
    print(f"  null mean             : {ci['mean']:.4f}")
    print(f"  null 95% CI           : [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    print()


# ---------------------------------------------------------------------------
# Example 3: Step-response metrics on a simulated step signal
# ---------------------------------------------------------------------------

def example_step_metrics():
    n = 300
    t = [i * (1.0 / 100.0) for i in range(n)]   # 100 Hz sample rate
    # Step at t=0.5 s (sample 50), with a modest overshoot
    y = []
    for i in range(n):
        if i < 50:
            y.append(0.0)
        elif i < 80:
            y.append(1.0 + 0.15 * math.exp(-0.1 * (i - 50)))  # overshoot
        else:
            y.append(1.0)

    m = step_metrics(t, y)

    print("=== Example 3: step_metrics() on a simulated step response ===")
    print(f"  initial_slope  : {m['initial_slope']:.4f}")
    print(f"  rise_time      : {m['rise_time']:.4f} s")
    print(f"  overshoot      : {m['overshoot']:.4f}")
    print(f"  settling_time  : {m['settling_time']:.4f} s")
    print(f"  steady_state   : {m['steady_state']:.4f}")
    print()


if __name__ == "__main__":
    example_analyze_sine()
    example_null_and_bootstrap()
    example_step_metrics()
