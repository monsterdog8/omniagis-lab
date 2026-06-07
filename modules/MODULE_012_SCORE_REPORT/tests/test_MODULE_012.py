"""Tests for MODULE_012_SCORE_REPORT — extracted from tests/test_gpts_core.py::TestScoreReport."""
from __future__ import annotations

import pytest

from gpts_core.score_report import (
    extract_sections,
    extract_verdict,
    extract_confidence,
    score_audit_report,
    validate_case_json,
    score_fail_closed,
)


class TestScoreReport:
    def _minimal_report_md(self):
        return (
            "## 1. Title\nTest Report\n"
            "## 2. Executive Summary\nThe claim is weakly supported. Verdict: WEAKLY SUPPORTED\n"
            "confidence score: 5/10\n"
            "## 3. Detailed Analysis\nEvidence shows partial support. Test 1: observable. "
            "observation inference modeling theory interpretation.\n"
            "## 4. Weakness Taxonomy\nType: Scope\nStatus: Active\nSeverity: Moderate\n"
            "Probable Cause: X\nDiscriminant Test: Y\n"
            "## 5. Limitations\nThe analysis is limited by the data. Conclusions are uncertain and fragile.\n"
            "## 6. Recommendations\nCollect more data.\n"
            "## 7. Sources\nArtifact A.\n"
        )

    def test_extract_sections(self):
        md = "## Title\nT\n## Executive Summary\nES\n## Detailed Analysis\nDA\n"
        sections = extract_sections(md)
        assert isinstance(sections, dict)

    def test_extract_verdict(self):
        text = "Verdict: SUPPORTED\nOther text"
        v = extract_verdict(text, ["SUPPORTED", "UNSUPPORTED"])
        assert v == "SUPPORTED"

    def test_extract_confidence(self):
        c = extract_confidence("confidence score: 7.5/10")
        assert c is not None
        assert 0 <= c <= 10
        assert extract_confidence("no confidence here") is None

    def test_score_audit_report_range(self):
        result = score_audit_report(self._minimal_report_md())
        assert isinstance(result, dict)
        total = result.get("total", result.get("score", 0))
        assert 0 <= total <= 100

    def test_score_audit_report_empty(self):
        result = score_audit_report("")
        assert isinstance(result, dict)

    def test_validate_report_structure_valid(self):
        case = {
            "case_id": "c1", "title": "T", "claim": "C", "question": "Q",
            "dossier": "D", "oracle": {"global_verdict": "SUPPORTED"},
            "scoring": {"components": {"a": 50, "b": 50}},
            "report_contract": {},
        }
        issues = validate_case_json(case)
        assert isinstance(issues, list)

    def test_score_fail_closed(self):
        sections = {"Limitations": "This analysis is limited. Conclusions are uncertain and insufficient."}
        score, note = score_fail_closed(sections)
        assert 0 <= score <= 10
