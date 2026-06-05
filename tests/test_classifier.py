"""Dedicated tests for FailClosedClassifier — the security-critical invariant."""
from __future__ import annotations

import pytest

from omniagis.core.classifier import FailClosedClassifier, VALID_VERDICTS


@pytest.fixture
def clf():
    return FailClosedClassifier()


# ---------------------------------------------------------------------------
# Fail-closed invariants
# ---------------------------------------------------------------------------

class TestFailClosedInvariant:
    def test_empty_list_is_no_pass(self, clf):
        assert clf.combine([]) == "NO PASS"

    def test_unknown_verdict_is_no_pass(self, clf):
        assert clf.combine(["PASS", "UNKNOWN_VERDICT"]) == "NO PASS"

    def test_empty_string_is_no_pass(self, clf):
        assert clf.combine([""]) == "NO PASS"

    def test_lowercase_pass_is_unknown_hence_no_pass(self, clf):
        assert clf.combine(["pass"]) == "NO PASS"

    def test_none_in_list_is_no_pass(self, clf):
        assert clf.combine([None]) == "NO PASS"  # type: ignore[list-item]

    def test_any_no_pass_dominates(self, clf):
        assert clf.combine(["PASS", "PARTIAL PASS", "NO PASS"]) == "NO PASS"

    def test_single_no_pass_dominates(self, clf):
        assert clf.combine(["NO PASS"]) == "NO PASS"

    def test_no_pass_with_all_other_pass(self, clf):
        assert clf.combine(["PASS"] * 10 + ["NO PASS"]) == "NO PASS"


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------

class TestPass:
    def test_single_pass(self, clf):
        assert clf.combine(["PASS"]) == "PASS"

    def test_multiple_pass(self, clf):
        assert clf.combine(["PASS", "PASS", "PASS"]) == "PASS"

    def test_large_all_pass(self, clf):
        assert clf.combine(["PASS"] * 100) == "PASS"


# ---------------------------------------------------------------------------
# PARTIAL PASS
# ---------------------------------------------------------------------------

class TestPartialPass:
    def test_single_partial(self, clf):
        assert clf.combine(["PARTIAL PASS"]) == "PARTIAL PASS"

    def test_partial_with_passes(self, clf):
        assert clf.combine(["PASS", "PARTIAL PASS", "PASS"]) == "PARTIAL PASS"

    def test_multiple_partials(self, clf):
        assert clf.combine(["PARTIAL PASS", "PARTIAL PASS"]) == "PARTIAL PASS"

    def test_partial_does_not_override_no_pass(self, clf):
        assert clf.combine(["PARTIAL PASS", "NO PASS"]) == "NO PASS"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_unknown_beats_no_pass(self, clf):
        # Both produce NO PASS — unknown is caught first but result is same
        assert clf.combine(["NO PASS", "GARBAGE"]) == "NO PASS"

    def test_no_pass_beats_partial(self, clf):
        assert clf.combine(["PARTIAL PASS", "NO PASS"]) == "NO PASS"

    def test_partial_beats_pass(self, clf):
        assert clf.combine(["PASS", "PARTIAL PASS"]) == "PARTIAL PASS"

    def test_all_three_valid_returns_no_pass(self, clf):
        assert clf.combine(["PASS", "PARTIAL PASS", "NO PASS"]) == "NO PASS"


# ---------------------------------------------------------------------------
# VALID_VERDICTS constant
# ---------------------------------------------------------------------------

class TestValidVerdicts:
    def test_valid_verdicts_contains_three(self):
        assert len(VALID_VERDICTS) == 3

    def test_valid_verdicts_members(self):
        assert VALID_VERDICTS == {"PASS", "PARTIAL PASS", "NO PASS"}

    def test_each_valid_verdict_alone_does_not_trigger_unknown_branch(self, clf):
        for v in VALID_VERDICTS:
            result = clf.combine([v])
            assert result in VALID_VERDICTS
