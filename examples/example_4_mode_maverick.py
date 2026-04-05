"""GO OMNIÆGIS Example 4: Mode MAVERICK Code Audit.

This example demonstrates how to use the Mode MAVERICK cold-pass auditor
to analyze a Python project and generate a comprehensive audit report.
"""

import sys
import tempfile
import textwrap
from pathlib import Path

from omniagis import ColdPass


def create_sample_project(root: Path):
    """Create a sample Python project for auditing."""
    # Good Python file
    (root / "main.py").write_text(
        textwrap.dedent("""
        import numpy as np
        from utils import calculate_mean

        def main():
            data = np.array([1, 2, 3, 4, 5])
            mean = calculate_mean(data)
            print(f"Mean: {mean}")

        if __name__ == "__main__":
            main()
        """)
    )

    # Utility module
    (root / "utils.py").write_text(
        textwrap.dedent("""
        def calculate_mean(data):
            return sum(data) / len(data)

        def calculate_std(data):
            mean = calculate_mean(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
        """)
    )

    # Documentation
    (root / "README.md").write_text(
        textwrap.dedent("""
        # Sample Project

        This is a sample project for demonstrating GO OMNIÆGIS Mode MAVERICK.

        ## Features
        - Data analysis utilities
        - Numpy integration
        """)
    )

    # Configuration
    (root / "pyproject.toml").write_text(
        textwrap.dedent("""
        [build-system]
        requires = ["setuptools>=68"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "sample-project"
        version = "0.1.0"
        """)
    )

    # Requirements
    (root / "requirements.txt").write_text("numpy>=1.24.0\n")

    print(f"✓ Sample project created at: {root}")


def main():
    print("=" * 70)
    print("GO OMNIÆGIS — Mode MAVERICK Code Audit Example")
    print("=" * 70)
    print()

    # Create temporary sample project
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        create_sample_project(root)
        print()

        # Run Mode MAVERICK audit
        print("Running Mode MAVERICK audit...")
        auditor = ColdPass()
        report = auditor.run(str(root))
        print()

        # Render full report
        print(auditor.render(report))
        print()

        # Extract key metrics
        print("=" * 70)
        print("KEY METRICS SUMMARY:")
        print("=" * 70)

        for entry in report.scorecard_entries:
            status_symbol = {
                "PASS": "✓",
                "PARTIAL PASS": "~",
                "NO PASS": "✗",
            }.get(entry.status, "?")
            print(f"  {status_symbol} {entry.metric_id}: {entry.name}")
            print(f"      Status: {entry.status}")
            print(f"      Detail: {entry.detail}")

        print()
        print("=" * 70)
        print(f"GLOBAL VERDICT: {report.global_verdict}")
        print("=" * 70)

        # Exit with appropriate code
        exit_code = {"PASS": 0, "PARTIAL PASS": 1, "NO PASS": 2}.get(
            report.global_verdict, 2
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
