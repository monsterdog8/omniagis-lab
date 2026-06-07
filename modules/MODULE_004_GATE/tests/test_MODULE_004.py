"""Tests for MODULE_004_GATE.

Extracted from tests/test_gpts_core.py::TestGate.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestGate:
    def test_classify_claim_blocked_production_ready(self):
        from gate import classify_claim
        result = classify_claim("This system is production-ready and approved for production")
        assert result.status == "BLOCKED"

    def test_classify_claim_blocked_sota(self):
        from gate import classify_claim
        result = classify_claim("Our model is SOTA and beats all benchmarks")
        assert result.status == "BLOCKED"

    def test_classify_claim_prudent_allowed(self):
        from gate import classify_claim
        result = classify_claim("This is a hypothesis, might be testable as LAB_ONLY prototype")
        assert result.status in ("ALLOWED_BOUNDED", "UNKNOWN", "UNKNOWN_REQUIRES_REVIEW")

    def test_classify_claim_has_status(self):
        from gate import classify_claim
        result = classify_claim("some claim text")
        assert hasattr(result, "status")
        assert result.status in ("BLOCKED", "ALLOWED_BOUNDED", "UNKNOWN", "UNKNOWN_REQUIRES_REVIEW")

    def test_evidence_score_high(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=10, present=10, valid=10, strict_sidecars=0, safety=1.0,
                               scoring=True, replay=True, independence=True)
        score = compute_evidence_score(inp)
        assert score.final_score >= 0.85
        assert hasattr(score, "public_claim_allowed")

    def test_evidence_score_low(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=10, present=2, valid=1, strict_sidecars=0, safety=0.3)
        score = compute_evidence_score(inp)
        assert score.final_score < 0.85
        assert not score.public_claim_allowed

    def test_evidence_score_zero_expected(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=0, present=0, valid=0)
        score = compute_evidence_score(inp)
        assert score.final_score == 0.0

    def test_proof_firewall_blocks_missing_dims(self):
        from gate import proof_firewall
        result = proof_firewall({"DATA": 0.0, "RAW": 0.5, "SCORING": 0.6,
                                 "REPLAY": 0.7, "INDEPENDENCE": 0.8, "SAFETY": 0.9})
        assert "BLOCKED" in result["verdict"]

    def test_proof_firewall_passes_all_dims(self):
        from gate import proof_firewall
        result = proof_firewall({"DATA": 0.8, "RAW": 0.8, "SCORING": 0.8,
                                 "REPLAY": 0.8, "INDEPENDENCE": 0.8, "SAFETY": 0.8})
        assert "BLOCKED" not in result["verdict"]

    def test_maturity_map_returns_dict(self):
        from gate import maturity_map
        result = maturity_map(has_raw=True, has_scoring=True, has_replay=False)
        assert isinstance(result, dict)
        assert "highest_maturity" in result or "stage" in result

    # Additional coverage tests

    def test_classify_claim_blocked_global_superiority(self):
        from gate import classify_claim
        result = classify_claim("Our system achieves global superiority over all competitors")
        assert result.status == "BLOCKED"
        assert "GLOBAL_SUPERIORITY" in result.hits

    def test_classify_claim_blocked_consciousness(self):
        from gate import classify_claim
        result = classify_claim("consciousness proven in this system")
        assert result.status == "BLOCKED"

    def test_classify_claim_unknown(self):
        from gate import classify_claim
        result = classify_claim("The output quality is generally acceptable")
        assert result.status == "UNKNOWN_REQUIRES_REVIEW"
        assert result.hits == []

    def test_classify_claim_to_dict(self):
        from gate import classify_claim
        result = classify_claim("hypothesis test")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "hits" in d

    def test_evidence_score_to_dict(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=5, present=5, valid=5, safety=0.8)
        score = compute_evidence_score(inp)
        d = score.to_dict()
        assert isinstance(d, dict)
        assert "final_score" in d
        assert "dimensions" in d

    def test_evidence_weights_sum_to_one(self):
        from gate import EVIDENCE_WEIGHTS
        total = sum(EVIDENCE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_maturity_stages_list(self):
        from gate import MATURITY_STAGES
        assert isinstance(MATURITY_STAGES, list)
        assert len(MATURITY_STAGES) == 8
        assert MATURITY_STAGES[0] == "IDEA"
        assert MATURITY_STAGES[-1] == "DEPLOY"

    def test_maturity_map_public_claim_always_blocked(self):
        from gate import maturity_map
        result = maturity_map(idea=True, design=True, prototype=True, raw=True,
                              scoring=True, replay=True, review=True, deploy=True)
        assert result["public_claim_right"] == "BLOCKED"
        assert result["production_status"] == "LOCKED"

    def test_proof_firewall_blocks_low_safety(self):
        from gate import proof_firewall
        result = proof_firewall({"DATA": 0.9, "RAW": 0.9, "SCORING": 0.9,
                                 "REPLAY": 0.9, "INDEPENDENCE": 0.9, "SAFETY": 0.5})
        assert "BLOCKED" in result["verdict"]

    def test_proof_firewall_none_dims(self):
        from gate import proof_firewall
        result = proof_firewall(None)
        assert "BLOCKED" in result["verdict"]
        assert len(result["missing"]) > 0

    def test_from_gate_counts_full_valid(self):
        from gate import from_gate_counts
        inp = from_gate_counts(expected=10, present=10, valid=10, safety=0.9,
                               scoring=True, replay=True, independence=True)
        assert inp.DATA == 1.0
        assert inp.RAW == 1.0
        assert inp.SCORING == 1.0
        assert inp.REPLAY == 1.0
        assert inp.INDEPENDENCE == 1.0

    def test_from_gate_counts_zero_expected(self):
        from gate import from_gate_counts
        inp = from_gate_counts(expected=0, present=0, valid=0)
        assert inp.DATA == 0.0
        assert inp.RAW == 0.0

    def test_evidence_score_scoring_status_ready(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=5, present=5, valid=5, safety=0.8, scoring=False)
        score = compute_evidence_score(inp)
        assert score.scoring_status == "READY_FOR_SCORING"

    def test_evidence_score_scoring_status_done(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=5, present=5, valid=5, safety=0.8, scoring=True)
        score = compute_evidence_score(inp)
        assert score.scoring_status == "SCORING_DONE_LOCAL"

    def test_evidence_score_blocked_incomplete_raw(self):
        from gate import from_gate_counts, compute_evidence_score
        inp = from_gate_counts(expected=10, present=10, valid=3, safety=0.8)
        score = compute_evidence_score(inp)
        assert score.scoring_status == "BLOCKED"

    def test_claim_status_enum_values(self):
        from gate import ClaimStatus
        assert ClaimStatus.BLOCKED.value == "BLOCKED"
        assert ClaimStatus.ALLOWED_BOUNDED.value == "ALLOWED_BOUNDED"
        assert ClaimStatus.UNKNOWN.value == "UNKNOWN_REQUIRES_REVIEW"
