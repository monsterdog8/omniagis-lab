"""Minimal usage examples for MODULE_002_CLASSIFIER.

Run from repository root:
    python modules/MODULE_002_CLASSIFIER/examples/example.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from classifier import classify_metric, LABELS

# ---------------------------------------------------------------------------
# Example 1: Classify a set of metric rows covering multiple label types
# ---------------------------------------------------------------------------

def example_classify_variety():
    rows = [
        ("luminance_mean",    "128.5",  "measured from image histogram"),
        ("result_ratio",      "0.72",   "mean of computed scores"),
        ("threshold_value",   "0.5",    "config parameter"),
        ("noise_sample",      "0.33",   "np.random randn synthetic generation"),
        ("Omega_constant",    "∞",      "cosmic singularity entity"),
        ("production_claim",  "ready",  "production-ready global superiority"),
    ]

    print("=== Example 1: classify_metric() across label types ===")
    print(f"  {'metric_name':<22} {'value_raw':<12} {'label':<12} {'conf':<6}  reason")
    print("  " + "-" * 75)
    for name, val, ctx in rows:
        label, reasons, conf = classify_metric(name, val, ctx)
        print(f"  {name:<22} {val:<12} {label:<12} {conf:.2f}   {reasons[0] if reasons else ''}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Equation row types
# ---------------------------------------------------------------------------

def example_equation_rows():
    print("=== Example 2: equation / recalculation row_type ===")

    # Recalculable arithmetic
    label, reasons, conf = classify_metric(
        "half_ratio", "0.5",
        row_type="recalculation",
        recalculable="True",
        result="0.5",
    )
    print(f"  recalculation (recalculable=True) -> label={label!r}, conf={conf:.2f}")

    # Non-recalculable equation
    label, reasons, conf = classify_metric(
        "formula_xy", "x + y",
        row_type="equation",
        recalculable=None,
        result=None,
    )
    print(f"  equation      (recalculable=None) -> label={label!r}, conf={conf:.2f}")

    # Symbolic equation
    label, reasons, conf = classify_metric(
        "Psi_value", "∑∞",
        context="quantum consciousness fractal",
        row_type="equation",
    )
    print(f"  equation      (symbolic context)  -> label={label!r}, conf={conf:.2f}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Bulk classification with label distribution summary
# ---------------------------------------------------------------------------

def example_bulk_classification():
    from collections import Counter

    metrics = [
        ("pixel_r",      "255",   "image pixel rgb measurement"),
        ("pixel_g",      "128",   "image pixel rgb measurement"),
        ("mask_dark",    "0.12",  "dark_ratio mask"),
        ("mean_score",   "0.68",  "mean computed via sum/count"),
        ("std_score",    "0.04",  "std variance computed"),
        ("sha256_hash",  "abc",   "sha256 hash computed"),
        ("status",       "PASS",  "verdict config status"),
        ("version",      "1.0.0", "reported config version"),
        ("rows",         "1024",  "rows generated reported"),
        ("benchmark_n",  "500",   "benchmark_rows simulation randn"),
        ("seed_val",     "42",    "random seed simulation"),
        ("x",            "1.0",   ""),
    ]

    counts: Counter = Counter()
    print("=== Example 3: Bulk classification and distribution ===")
    for name, val, ctx in metrics:
        label, _, _ = classify_metric(name, val, ctx)
        counts[label] += 1

    for lab in LABELS:
        print(f"  {lab:<12}: {counts[lab]}")
    print()


if __name__ == "__main__":
    example_classify_variety()
    example_equation_rows()
    example_bulk_classification()
