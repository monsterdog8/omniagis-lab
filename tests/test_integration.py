"""Integration tests for GO OMNIÆGIS — full framework capability demonstration."""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from omniagis import (
    EpsilonRobustnessValidator,
    ReturnTimeStatistics,
    FailClosedClassifier,
    ColdPass,
)


class TestGOOmniAegisCore:
    """Test GO OMNIÆGIS core validation components integration."""

    def test_epsilon_robustness_pass(self):
        """Verify ε-robustness validator accepts identical trajectories."""
        validator = EpsilonRobustnessValidator(epsilon=1e-3)
        trajectory = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        reference = trajectory.copy()
        result = validator.validate(trajectory, reference)
        assert result.status == "PASS"
        assert result.max_dist < 1e-10
        assert result.mean_dist < 1e-10

    def test_epsilon_robustness_partial_pass(self):
        """Verify ε-robustness validator handles moderate deviations."""
        validator = EpsilonRobustnessValidator(epsilon=0.5)
        trajectory = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        reference = np.array([[1.0, 2.0], [3.0, 4.0], [5.8, 6.8]])
        result = validator.validate(trajectory, reference)
        assert result.status == "PARTIAL PASS"
        assert result.mean_dist <= 0.5
        assert result.max_dist > 0.5

    def test_return_time_statistics_periodic(self):
        """Verify return-time statistics on periodic signal."""
        rts = ReturnTimeStatistics(tolerance=0.1)
        # Sine wave with period ~6.28 (2π)
        t = np.linspace(0, 20, 200)
        series = np.sin(t)
        indices = rts.find_returns(series, target_value=0.0)
        assert len(indices) >= 6  # Multiple returns to zero
        stats = rts.compute_stats(indices)
        assert stats["mean"] > 0
        assert stats["count"] >= 6

    def test_fail_closed_classifier_all_pass(self):
        """Verify fail-closed classifier with all PASS verdicts."""
        clf = FailClosedClassifier()
        verdict = clf.combine(["PASS", "PASS", "PASS"])
        assert verdict == "PASS"

    def test_fail_closed_classifier_partial_pass(self):
        """Verify fail-closed classifier with mixed verdicts."""
        clf = FailClosedClassifier()
        verdict = clf.combine(["PASS", "PARTIAL PASS", "PASS"])
        assert verdict == "PARTIAL PASS"

    def test_fail_closed_classifier_no_pass(self):
        """Verify fail-closed classifier with NO PASS present."""
        clf = FailClosedClassifier()
        verdict = clf.combine(["PASS", "PARTIAL PASS", "NO PASS"])
        assert verdict == "NO PASS"


class TestGOOmniAegisAudit:
    """Test GO OMNIÆGIS audit components (Mode MAVERICK) integration."""

    def test_cold_pass_on_clean_project(self):
        """Verify Mode MAVERICK audit on a clean minimal project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a minimal clean Python project
            (root / "main.py").write_text(
                textwrap.dedent("""
                def hello():
                    return "world"

                if __name__ == "__main__":
                    print(hello())
                """)
            )
            (root / "utils.py").write_text(
                textwrap.dedent("""
                def add(a, b):
                    return a + b
                """)
            )
            (root / "README.md").write_text("# Test Project\n")
            (root / "requirements.txt").write_text("numpy\n")

            # Run Mode MAVERICK audit
            auditor = ColdPass()
            report = auditor.run(str(root))

            # Verify report structure
            assert report.global_verdict in ["PASS", "PARTIAL PASS", "NO PASS"]
            assert report.path == str(root)
            assert len(report.structured_inventory) > 0
            assert len(report.scorecard) > 0
            assert len(report.file_audit) > 0

            # Verify scorecard entries
            assert len(report.scorecard_entries) == 12  # M1–M12
            metric_ids = [e.metric_id for e in report.scorecard_entries]
            assert metric_ids == [
                "M1", "M2", "M3", "M4", "M5", "M6",
                "M7", "M8", "M9", "M10", "M11", "M12"
            ]

            # Verify inventory
            assert report.inventory_report is not None
            assert len(report.inventory_report.files) >= 4  # 2 .py + README + requirements

            # Verify parse results
            assert len(report.parse_results) >= 2  # main.py + utils.py
            for res in report.parse_results:
                assert res.parseable is True

    def test_cold_pass_detects_syntax_errors(self):
        """Verify Mode MAVERICK audit detects syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create file with syntax error
            (root / "broken.py").write_text("def bad_syntax(\n")

            # Run audit
            auditor = ColdPass()
            report = auditor.run(str(root))

            # Verify NO PASS verdict
            assert report.global_verdict == "NO PASS"

            # Verify syntax error detected in parse results
            broken_result = next(
                (r for r in report.parse_results if "broken.py" in r.path), None
            )
            assert broken_result is not None
            assert broken_result.parseable is False
            assert broken_result.syntax_error is not None

    def test_cold_pass_detects_duplicates(self):
        """Verify Mode MAVERICK audit detects duplicate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create duplicate files with identical content
            content = "def duplicate(): pass\n"
            (root / "file1.py").write_text(content)
            (root / "file2.py").write_text(content)

            # Run audit
            auditor = ColdPass()
            report = auditor.run(str(root))

            # Verify duplicates detected
            assert len(report.inventory_report.duplicates) >= 1

            # Verify M3 metric reports duplicates
            m3_entry = next(
                (e for e in report.scorecard_entries if e.metric_id == "M3"), None
            )
            assert m3_entry is not None
            assert m3_entry.status in ["PARTIAL PASS", "NO PASS"]


class TestGOOmniAegisFullPipeline:
    """Test complete GO OMNIÆGIS pipeline: validation + audit."""

    def test_validate_then_audit_workflow(self):
        """Verify full GO OMNIÆGIS workflow: validate dynamics, then audit code."""
        # Step 1: Validate dynamical system trajectory
        validator = EpsilonRobustnessValidator(epsilon=1e-2)
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        reference = np.array([[0.0, 0.0], [1.01, 1.01], [2.0, 2.0]])
        validation_result = validator.validate(trajectory, reference)
        assert validation_result.status in ["PASS", "PARTIAL PASS"]

        # Step 2: Compute return-time statistics
        rts = ReturnTimeStatistics(tolerance=0.05)
        t = np.linspace(0, 10, 100)
        series = np.sin(2 * np.pi * t)
        indices = rts.find_returns(series, target_value=0.0)
        stats = rts.compute_stats(indices)
        verdict = rts.classify(stats, max_allowed_mean=30)
        assert verdict in ["PASS", "PARTIAL PASS", "NO PASS"]

        # Step 3: Combine verdicts with fail-closed logic
        clf = FailClosedClassifier()
        global_verdict = clf.combine([validation_result.status, verdict])
        assert global_verdict in ["PASS", "PARTIAL PASS", "NO PASS"]

        # Step 4: Audit the repository with Mode MAVERICK
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "analysis.py").write_text(
                textwrap.dedent("""
                import numpy as np

                def analyze_trajectory(data):
                    return np.mean(data)
                """)
            )

            auditor = ColdPass()
            report = auditor.run(str(root))
            audit_verdict = report.global_verdict
            assert audit_verdict in ["PASS", "PARTIAL PASS", "NO PASS"]

            # Step 5: Final fail-closed verdict combining all results
            final_verdict = clf.combine([global_verdict, audit_verdict])
            assert final_verdict in ["PASS", "PARTIAL PASS", "NO PASS"]

    def test_public_api_imports(self):
        """Verify all GO OMNIÆGIS components available via public API."""
        # All imports should work from top-level package
        from omniagis import (
            __version__,
            EpsilonRobustnessValidator,
            ReturnTimeStatistics,
            FailClosedClassifier,
            FileInventory,
            ParsabilityChecker,
            build_scorecard,
            ColdPass,
        )

        assert __version__ == "0.1.0"
        assert EpsilonRobustnessValidator is not None
        assert ReturnTimeStatistics is not None
        assert FailClosedClassifier is not None
        assert FileInventory is not None
        assert ParsabilityChecker is not None
        assert build_scorecard is not None
        assert ColdPass is not None
