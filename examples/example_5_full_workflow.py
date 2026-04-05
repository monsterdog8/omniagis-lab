"""GO OMNIÆGIS Example 5: Complete Workflow Integration.

This example demonstrates the complete GO OMNIÆGIS workflow combining:
- Epsilon-robustness validation
- Return-time statistics
- Fail-closed classification
- Mode MAVERICK audit

This showcases the full capability of the framework.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np

from omniagis import (
    EpsilonRobustnessValidator,
    ReturnTimeStatistics,
    FailClosedClassifier,
    ColdPass,
)


def main():
    print("=" * 70)
    print("GO OMNIÆGIS — COMPLETE WORKFLOW INTEGRATION")
    print("=" * 70)
    print()

    # =======================================================================
    # PHASE 1: Dynamical System Validation
    # =======================================================================
    print("PHASE 1: DYNAMICAL SYSTEM VALIDATION")
    print("-" * 70)

    # Simulate a dynamical system trajectory
    print("Simulating numerical trajectory...")
    t = np.linspace(0, 10, 100)
    trajectory = np.column_stack([np.sin(t), np.cos(t)])
    reference = np.column_stack([np.sin(t) * 1.01, np.cos(t) * 0.99])
    print(f"  Trajectory shape: {trajectory.shape}")
    print()

    # Validate with ε-robustness
    print("Validating trajectory with ε-robustness...")
    validator = EpsilonRobustnessValidator(epsilon=0.05)
    validation_result = validator.validate(trajectory, reference)
    print(f"  Epsilon: {validator.epsilon}")
    print(f"  Max distance: {validation_result.max_dist:.6f}")
    print(f"  Mean distance: {validation_result.mean_dist:.6f}")
    print(f"  Verdict: {validation_result.status}")
    print()

    # =======================================================================
    # PHASE 2: Return-Time Analysis (Poincaré Recurrence)
    # =======================================================================
    print("PHASE 2: RETURN-TIME ANALYSIS")
    print("-" * 70)

    # Analyze recurrence behavior
    print("Computing return-time statistics...")
    rts = ReturnTimeStatistics(tolerance=0.05)
    series = trajectory[:, 0]  # First component (sine wave)
    indices = rts.find_returns(series, target_value=0.0)
    stats = rts.compute_stats(indices)
    recurrence_verdict = rts.classify(stats, max_allowed_mean=30, min_required_returns=3)

    print(f"  Tolerance: {rts.tolerance}")
    print(f"  Returns found: {stats['count']}")
    print(f"  Mean return time: {stats['mean']:.2f}")
    print(f"  Verdict: {recurrence_verdict}")
    print()

    # =======================================================================
    # PHASE 3: Fail-Closed Aggregation
    # =======================================================================
    print("PHASE 3: FAIL-CLOSED VERDICT AGGREGATION")
    print("-" * 70)

    clf = FailClosedClassifier()
    dynamics_verdict = clf.combine([validation_result.status, recurrence_verdict])
    print(f"  ε-robustness verdict: {validation_result.status}")
    print(f"  Return-time verdict: {recurrence_verdict}")
    print(f"  → Combined dynamics verdict: {dynamics_verdict}")
    print()

    # =======================================================================
    # PHASE 4: Mode MAVERICK Code Audit
    # =======================================================================
    print("PHASE 4: MODE MAVERICK CODE AUDIT")
    print("-" * 70)

    # Create a sample codebase for auditing
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create simulation code
        (root / "simulator.py").write_text(
            textwrap.dedent("""
            import numpy as np

            def simulate_trajectory(t_max=10, n_steps=100):
                t = np.linspace(0, t_max, n_steps)
                x = np.sin(t)
                y = np.cos(t)
                return np.column_stack([x, y])

            def validate_trajectory(trajectory, reference, epsilon=0.05):
                diff = trajectory - reference
                distances = np.sqrt(np.sum(diff**2, axis=1))
                max_dist = np.max(distances)
                mean_dist = np.mean(distances)

                if max_dist <= epsilon:
                    return "PASS"
                elif mean_dist <= epsilon:
                    return "PARTIAL PASS"
                else:
                    return "NO PASS"
            """)
        )

        (root / "analyzer.py").write_text(
            textwrap.dedent("""
            def compute_statistics(data):
                return {
                    "mean": sum(data) / len(data),
                    "min": min(data),
                    "max": max(data),
                }
            """)
        )

        (root / "README.md").write_text("# Trajectory Analysis Project\n")
        (root / "requirements.txt").write_text("numpy>=1.24.0\n")

        print("Running comprehensive code audit...")
        auditor = ColdPass()
        audit_report = auditor.run(str(root))

        print(f"  Files audited: {len(audit_report.inventory_report.files)}")
        print(f"  Python files: {len(audit_report.parse_results)}")

        # Show M1-M12 scorecard summary
        pass_count = sum(1 for e in audit_report.scorecard_entries if e.status == "PASS")
        partial_count = sum(1 for e in audit_report.scorecard_entries if e.status == "PARTIAL PASS")
        fail_count = sum(1 for e in audit_report.scorecard_entries if e.status == "NO PASS")

        print(f"  Scorecard M1-M12: {pass_count} PASS, {partial_count} PARTIAL, {fail_count} FAIL")
        print(f"  Audit verdict: {audit_report.global_verdict}")
        print()

        # =======================================================================
        # PHASE 5: Final System-Wide Verdict
        # =======================================================================
        print("PHASE 5: FINAL SYSTEM-WIDE VERDICT")
        print("-" * 70)

        final_verdict = clf.combine([dynamics_verdict, audit_report.global_verdict])

        print("Component verdicts:")
        print(f"  • Dynamics validation: {dynamics_verdict}")
        print(f"  • Code audit: {audit_report.global_verdict}")
        print()
        print(f"→ FINAL GO OMNIÆGIS VERDICT: {final_verdict}")
        print()

    # =======================================================================
    # Summary
    # =======================================================================
    print("=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print("GO OMNIÆGIS provides comprehensive validation for:")
    print("  1. Dynamical system trajectories (ε-robustness)")
    print("  2. Temporal recurrence patterns (Poincaré statistics)")
    print("  3. Multi-component verdict aggregation (fail-closed logic)")
    print("  4. Code quality assurance (Mode MAVERICK M1-M12)")
    print()
    print("This ensures RIGOROUS, FAIL-CLOSED validation across all aspects")
    print("of computational scientific workflows.")
    print("=" * 70)


if __name__ == "__main__":
    main()
