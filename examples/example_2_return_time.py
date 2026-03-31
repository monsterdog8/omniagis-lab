"""GO OMNIÆGIS Example 2: Return-Time Statistics (Poincaré Recurrence).

This example demonstrates how to compute Poincaré recurrence statistics
on scalar time series using return-time analysis.
"""

import numpy as np
from omniagis import ReturnTimeStatistics


def main():
    print("=" * 70)
    print("GO OMNIÆGIS — Return-Time Statistics Example")
    print("=" * 70)
    print()

    # Create return-time analyzer with tolerance = 0.1
    rts = ReturnTimeStatistics(tolerance=0.1)
    print(f"ReturnTimeStatistics configured with tolerance = {rts.tolerance}")
    print()

    # Example 1: Periodic sine wave
    print("Example 1: Periodic sine wave (period ≈ 2π)")
    t = np.linspace(0, 20, 200)
    series_1 = np.sin(t)
    target_value = 0.0

    indices_1 = rts.find_returns(series_1, target_value=target_value)
    stats_1 = rts.compute_stats(indices_1)
    verdict_1 = rts.classify(stats_1, max_allowed_mean=40, min_required_returns=5)

    print(f"  Target value: {target_value}")
    print(f"  Number of returns: {stats_1['count']}")
    print(f"  Mean return time: {stats_1['mean']:.2f} timesteps")
    print(f"  Std deviation: {stats_1['std']:.2f}")
    print(f"  Max return time: {stats_1['max']:.0f}")
    print(f"  Min return time: {stats_1['min']:.0f}")
    print(f"  Verdict: {verdict_1}")
    print()

    # Example 2: Damped oscillation
    print("Example 2: Damped oscillation")
    t2 = np.linspace(0, 20, 200)
    series_2 = np.exp(-0.1 * t2) * np.sin(2 * np.pi * t2)

    indices_2 = rts.find_returns(series_2, target_value=0.0)
    stats_2 = rts.compute_stats(indices_2)
    verdict_2 = rts.classify(stats_2, max_allowed_mean=20, min_required_returns=5)

    print(f"  Number of returns: {stats_2['count']}")
    print(f"  Mean return time: {stats_2['mean']:.2f} timesteps")
    print(f"  Verdict: {verdict_2}")
    print()

    # Example 3: Chaotic-like signal (random walk)
    print("Example 3: Random walk (chaotic-like behavior)")
    np.random.seed(42)
    series_3 = np.cumsum(np.random.randn(500) * 0.1)

    indices_3 = rts.find_returns(series_3, target_value=0.0)
    stats_3 = rts.compute_stats(indices_3)
    verdict_3 = rts.classify(stats_3, max_allowed_mean=50, min_required_returns=3)

    print(f"  Number of returns: {stats_3['count']}")
    if stats_3['count'] > 0:
        print(f"  Mean return time: {stats_3['mean']:.2f} timesteps")
        print(f"  Std deviation: {stats_3['std']:.2f}")
    else:
        print(f"  Mean return time: N/A (no returns found)")
    print(f"  Verdict: {verdict_3}")
    print()

    print("=" * 70)
    print("Return-Time Analysis Summary:")
    print(f"  Tolerance: ±{rts.tolerance}")
    print(f"  A 'return' occurs when |series[i] - target| ≤ tolerance")
    print(f"  Return time = gap between successive returns")
    print(f"  PASS: mean ≤ threshold AND count ≥ min_required")
    print(f"  PARTIAL PASS: only one condition met")
    print(f"  NO PASS: neither condition met")
    print("=" * 70)


if __name__ == "__main__":
    main()
