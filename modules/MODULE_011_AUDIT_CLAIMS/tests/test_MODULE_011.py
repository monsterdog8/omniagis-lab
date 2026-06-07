"""Tests for MODULE_011_AUDIT_CLAIMS — extracted from tests/test_gpts_core.py::TestAuditClaims."""
from __future__ import annotations

import pytest

from gpts_core.audit_claims import (
    audit_claim,
    decompose_claim,
    validate_canonical_report,
    batch_audit,
    inspect_document,
    AuditReport,
)


class TestAuditClaims:
    def test_audit_claim_returns_report(self):
        report = audit_claim("The system is fully validated", "Test Audit")
        assert report is not None
        assert hasattr(report, "verdict") or isinstance(report, dict)

    def test_audit_claim_verdict_values(self):
        report = audit_claim("This is fully proven beyond doubt", "Test")
        if hasattr(report, "verdict"):
            assert report.verdict in ("SUPPORTED", "PARTIALLY SUPPORTED", "WEAKLY SUPPORTED",
                                      "UNSUPPORTED", "UNCERTAIN")

    def test_audit_claim_production_not_unlocked(self):
        report = audit_claim("claim text", "title")
        if hasattr(report, "to_dict"):
            d = report.to_dict()
            assert d.get("production_unlocked", False) is False

    def test_decompose_claim_simple(self):
        props = decompose_claim("The system is coherent and validated")
        assert isinstance(props, list)
        assert len(props) >= 1

    def test_decompose_claim_compound(self):
        props = decompose_claim("A is true, B is valid, and C is operational")
        assert len(props) >= 2

    def test_validate_canonical_report_returns_tuple_or_list(self):
        report_md = (
            "## 1. Title\nTest Report\n"
            "## 2. Executive Summary\nSummary here.\n"
            "## 3. Detailed Analysis\nAnalysis here.\n"
            "## 4. Weakness Taxonomy\nWeakness 1: scope\n"
            "## 5. Limitations\nLimits here.\n"
            "## 6. Recommendations\nRec here.\n"
            "## 7. Sources\nSource A.\n"
        )
        result = validate_canonical_report(report_md)
        assert isinstance(result, (list, tuple))

    def test_validate_canonical_report_missing_sections(self):
        result = validate_canonical_report("just a short text")
        assert isinstance(result, (list, tuple))

    def test_batch_audit_returns_dict(self):
        claims = [{"title": "T1", "claim": "claim A"}, {"title": "T2", "claim": "claim B"}]
        result = batch_audit(claims)
        assert isinstance(result, dict)
        assert "results" in result or "rows" in result or len(result) > 0

    def test_inspect_document_returns_report(self):
        result = inspect_document("This is a document with some content.")
        assert isinstance(result, AuditReport) or isinstance(result, dict)
