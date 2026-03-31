"""GO OMNIÆGIS Example 3: Fail-Closed Classification.

This example demonstrates how to aggregate multiple verdicts using
fail-closed logic.
"""

from omniagis import FailClosedClassifier


def main():
    print("=" * 70)
    print("GO OMNIÆGIS — Fail-Closed Classification Example")
    print("=" * 70)
    print()

    clf = FailClosedClassifier()
    print("FailClosedClassifier initialized")
    print("Logic: Any NO PASS → NO PASS")
    print("       Any PARTIAL PASS (no NO PASS) → PARTIAL PASS")
    print("       All PASS → PASS")
    print()

    # Example 1: All PASS
    print("Example 1: All verdicts are PASS")
    verdicts_1 = ["PASS", "PASS", "PASS", "PASS"]
    result_1 = clf.combine(verdicts_1)
    print(f"  Input: {verdicts_1}")
    print(f"  Output: {result_1}")
    print()

    # Example 2: Mixed PASS and PARTIAL PASS
    print("Example 2: Mixed PASS and PARTIAL PASS")
    verdicts_2 = ["PASS", "PARTIAL PASS", "PASS", "PASS"]
    result_2 = clf.combine(verdicts_2)
    print(f"  Input: {verdicts_2}")
    print(f"  Output: {result_2}")
    print()

    # Example 3: Contains NO PASS
    print("Example 3: Contains NO PASS (fail-closed behavior)")
    verdicts_3 = ["PASS", "PARTIAL PASS", "NO PASS", "PASS"]
    result_3 = clf.combine(verdicts_3)
    print(f"  Input: {verdicts_3}")
    print(f"  Output: {result_3}")
    print()

    # Example 4: Single NO PASS dominates
    print("Example 4: Single NO PASS dominates all")
    verdicts_4 = ["PASS", "PASS", "PASS", "PASS", "PASS", "NO PASS"]
    result_4 = clf.combine(verdicts_4)
    print(f"  Input: {verdicts_4}")
    print(f"  Output: {result_4}")
    print()

    # Example 5: Empty list
    print("Example 5: Empty verdict list (edge case)")
    verdicts_5 = []
    result_5 = clf.combine(verdicts_5)
    print(f"  Input: {verdicts_5}")
    print(f"  Output: {result_5}")
    print()

    # Example 6: Practical multi-component validation
    print("Example 6: Practical multi-component system validation")
    component_verdicts = {
        "Trajectory validation": "PASS",
        "Return-time analysis": "PASS",
        "Parsability check": "PASS",
        "Dependency check": "PARTIAL PASS",
        "Syntax validation": "PASS",
    }
    print(f"  Component verdicts:")
    for component, verdict in component_verdicts.items():
        print(f"    {component}: {verdict}")

    system_verdict = clf.combine(list(component_verdicts.values()))
    print(f"  System-wide verdict: {system_verdict}")
    print()

    print("=" * 70)
    print("Fail-Closed Logic Summary:")
    print("  This ensures CONSERVATIVE validation behavior:")
    print("  • Any failure → system fails")
    print("  • Any partial success → system partially succeeds")
    print("  • Only all successes → system succeeds")
    print("=" * 70)


if __name__ == "__main__":
    main()
