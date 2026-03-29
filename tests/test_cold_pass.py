"""Tests for ColdPass (Mode MAVERICK)."""

from __future__ import annotations

import os
import textwrap

import pytest

from omniagis.audit.cold_pass import ColdPass, ColdPassReport

# Locate repo root dynamically: tests/ lives one level below the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_VERDICTS = {"PASS", "PARTIAL PASS", "NO PASS"}


class TestColdPassOnRepoRoot:
    def test_run_returns_report(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        assert isinstance(report, ColdPassReport)

    def test_global_verdict_is_valid(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        assert report.global_verdict in VALID_VERDICTS

    def test_all_sections_populated(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        assert len(report.structured_inventory) > 0
        assert len(report.scorecard) > 0
        assert len(report.file_audit) > 0
        assert len(report.tensions) > 0
        assert len(report.cleanup_plan) > 0
        assert len(report.validation_plan) > 0
        assert len(report.minimal_core) > 0

    def test_scorecard_has_m1_through_m12(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        for i in range(1, 13):
            assert f"M{i}" in report.scorecard

    def test_render_non_empty(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        rendered = cp.render(report)
        assert isinstance(rendered, str)
        assert len(rendered) > 100
        assert "GLOBAL VERDICT" in rendered


class TestColdPassOnKnownDirectory:
    """Test ColdPass on a directory created within the repo."""

    _TEST_DIR = os.path.join(REPO_ROOT, "_test_cold_pass_tmp")

    def setup_method(self) -> None:
        os.makedirs(self._TEST_DIR, exist_ok=True)
        # Create a valid Python file
        with open(os.path.join(self._TEST_DIR, "good.py"), "w") as fh:
            fh.write(
                textwrap.dedent("""\
                    import os

                    def hello():
                        return "world"
                """)
            )
        # Create a doc file
        with open(os.path.join(self._TEST_DIR, "README.md"), "w") as fh:
            fh.write("# Test\n")

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self._TEST_DIR, ignore_errors=True)

    def test_run_on_known_dir(self) -> None:
        cp = ColdPass()
        report = cp.run(self._TEST_DIR)
        assert report.global_verdict in VALID_VERDICTS

    def test_inventory_finds_files(self) -> None:
        cp = ColdPass()
        report = cp.run(self._TEST_DIR)
        assert report.inventory_report is not None
        assert len(report.inventory_report.files) >= 2

    def test_parse_results_present(self) -> None:
        cp = ColdPass()
        report = cp.run(self._TEST_DIR)
        assert len(report.parse_results) >= 1

    def test_good_py_is_parseable(self) -> None:
        cp = ColdPass()
        report = cp.run(self._TEST_DIR)
        good = [r for r in report.parse_results if r.path.endswith("good.py")]
        assert good
        assert good[0].parseable


class TestRenderOutput:
    def test_all_section_headers_present(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        rendered = cp.render(report)
        for section in ["A.", "B.", "C.", "D.", "E.", "F.", "G."]:
            assert section in rendered

    def test_verdict_in_render(self) -> None:
        cp = ColdPass()
        report = cp.run(REPO_ROOT)
        rendered = cp.render(report)
        assert report.global_verdict in rendered
