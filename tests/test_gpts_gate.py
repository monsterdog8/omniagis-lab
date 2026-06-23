"""Tests for gpts_core.gate — claim classification and evidence scoring."""
from __future__ import annotations

import pytest

from gpts_core.gate import (
    classify_claim,
    compute_evidence_score,
    from_gate_counts,
    maturity_map,
    proof_firewall,
    EvidenceInput,
    ClaimStatus,
    EVIDENCE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# classify_claim
# ---------------------------------------------------------------------------

class TestClassifyClaim:
    def test_sota_is_blocked(self):
        result = classify_claim("Our model achieves SOTA on all benchmarks.")
        assert result.status == ClaimStatus.BLOCKED.value
        assert "BENCHMARK_WON" in result.hits

    def test_production_ready_is_blocked(self):
        result = classify_claim("The system is production-ready for deployment.")
        assert result.status == ClaimStatus.BLOCKED.value

    def test_scientific_validation_is_blocked(self):
        result = classify_claim("This is scientifically validated.")
        assert result.status == ClaimStatus.BLOCKED.value

    def test_hypothesis_is_bounded(self):
        result = classify_claim("This is a testable hypothesis in a LAB_ONLY setting.")
        assert result.status == ClaimStatus.ALLOWED_BOUNDED.value
        assert len(result.hits) > 0

    def test_evidence_workflow_is_bounded(self):
        result = classify_claim("We have raw data with hash and scoring.")
        assert result.status == ClaimStatus.ALLOWED_BOUNDED.value

    def test_neutral_text_is_unknown(self):
        result = classify_claim("The weather is nice today.")
        assert result.status == ClaimStatus.UNKNOWN.value
        assert result.hits == []

    def test_empty_string_is_unknown(self):
        result = classify_claim("")
        assert result.status == ClaimStatus.UNKNOWN.value

    def test_blocked_has_next_gate(self):
        result = classify_claim("beats GPT in every test")
        assert "RAW_COLLECTION" in result.required_next_gate

    def test_to_dict_contains_status(self):
        result = classify_claim("prototype simulation test")
        d = result.to_dict()
        assert "status" in d
        assert "hits" in d

    def test_case_insensitive(self):
        result = classify_claim("scientifically VALIDATED study")
        assert result.status == ClaimStatus.BLOCKED.value


# ---------------------------------------------------------------------------
# compute_evidence_score
# ---------------------------------------------------------------------------

class TestComputeEvidenceScore:
    def _minimal_pass_input(self) -> EvidenceInput:
        return EvidenceInput(
            DATA=1.0, RAW=1.0, SCORING=1.0, REPLAY=1.0,
            INDEPENDENCE=1.0, SAFETY=1.0,
            raw_expected=3, raw_valid=3,
            scoring_done=True, replay_available=True,
            independent_review=True, strong_public_claim=False,
        )

    def test_no_raw_outputs_penalized(self):
        inp = EvidenceInput(DATA=0.0, RAW=0.0)
        score = compute_evidence_score(inp)
        assert score.final_score == 0.0
        assert any("NO_RAW_OUTPUTS" in p for p in score.penalties)

    def test_raw_collection_incomplete_penalized(self):
        inp = EvidenceInput(
            DATA=1.0, RAW=1.0, raw_expected=5, raw_valid=3,
        )
        score = compute_evidence_score(inp)
        assert any("RAW_COLLECTION_INCOMPLETE" in p for p in score.penalties)

    def test_scoring_not_done_penalized(self):
        inp = EvidenceInput(DATA=1.0, RAW=1.0, raw_expected=1, raw_valid=1, scoring_done=False)
        score = compute_evidence_score(inp)
        assert any("SCORING_NOT_EXECUTED" in p for p in score.penalties)

    def test_no_independent_review_penalized(self):
        inp = EvidenceInput(DATA=1.0, RAW=1.0, INDEPENDENCE=0.0, independent_review=False)
        score = compute_evidence_score(inp)
        assert any("NO_INDEPENDENT_REVIEW" in p for p in score.penalties)

    def test_strong_claim_penalized(self):
        inp = EvidenceInput(DATA=1.0, RAW=1.0, strong_public_claim=True)
        score = compute_evidence_score(inp)
        assert any("STRONG_CLAIM" in p for p in score.penalties)

    def test_public_claim_not_allowed_by_default(self):
        inp = EvidenceInput()
        score = compute_evidence_score(inp)
        assert not score.public_claim_allowed

    def test_public_claim_allowed_requires_high_score(self):
        inp = self._minimal_pass_input()
        score = compute_evidence_score(inp)
        if score.final_score >= 0.85:
            assert score.public_claim_allowed

    def test_safety_below_half_penalized(self):
        inp = EvidenceInput(DATA=1.0, RAW=1.0, SAFETY=0.3)
        score = compute_evidence_score(inp)
        assert any("SAFETY_INSUFFICIENT" in p for p in score.penalties)

    def test_scoring_status_ready_when_raw_passes_scoring_not_done(self):
        inp = EvidenceInput(
            RAW=1.0, raw_expected=2, raw_valid=2, scoring_done=False,
        )
        score = compute_evidence_score(inp)
        assert score.scoring_status == "READY_FOR_SCORING"

    def test_scoring_status_done_local(self):
        inp = EvidenceInput(
            RAW=1.0, raw_expected=2, raw_valid=2, scoring_done=True,
        )
        score = compute_evidence_score(inp)
        assert score.scoring_status == "SCORING_DONE_LOCAL"

    def test_to_dict_has_verdict(self):
        inp = EvidenceInput()
        score = compute_evidence_score(inp)
        d = score.to_dict()
        assert "verdict" in d
        assert "final_score" in d

    def test_final_score_in_zero_one(self):
        for _ in range(5):
            inp = EvidenceInput(DATA=0.5, RAW=0.5, SCORING=0.5, REPLAY=0.5,
                                INDEPENDENCE=0.5, SAFETY=0.5)
            score = compute_evidence_score(inp)
            assert 0.0 <= score.final_score <= 1.0


# ---------------------------------------------------------------------------
# from_gate_counts
# ---------------------------------------------------------------------------

class TestFromGateCounts:
    def test_all_valid_gives_raw_one(self):
        inp = from_gate_counts(expected=5, present=5, valid=5, safety=0.8)
        assert inp.RAW == pytest.approx(1.0)
        assert inp.DATA == 1.0

    def test_zero_expected_gives_raw_zero(self):
        inp = from_gate_counts(expected=0, present=0, valid=0)
        assert inp.RAW == 0.0
        assert inp.DATA == 0.0

    def test_partial_valid_gives_partial_raw(self):
        inp = from_gate_counts(expected=4, present=4, valid=2)
        assert inp.RAW == pytest.approx(0.5)

    def test_scoring_flag_sets_scoring(self):
        inp = from_gate_counts(expected=3, present=3, valid=3, scoring=True)
        assert inp.SCORING == 1.0
        assert inp.scoring_done is True

    def test_replay_flag_sets_replay(self):
        inp = from_gate_counts(expected=2, present=2, valid=2, replay=True)
        assert inp.REPLAY == 1.0
        assert inp.replay_available is True

    def test_independence_flag(self):
        inp = from_gate_counts(expected=2, present=2, valid=2, independence=True)
        assert inp.INDEPENDENCE == 1.0
        assert inp.independent_review is True


# ---------------------------------------------------------------------------
# maturity_map
# ---------------------------------------------------------------------------

class TestMaturityMap:
    def test_no_flags_gives_none(self):
        result = maturity_map()
        assert result["highest_maturity"] == "NONE"

    def test_prototype_flag(self):
        result = maturity_map(idea=True, design=True, prototype=True)
        assert result["highest_maturity"] == "PROTOTYPE"

    def test_production_always_locked(self):
        result = maturity_map(idea=True, design=True, deploy=True)
        assert result["production_status"] == "LOCKED"
        assert result["public_claim_right"] == "BLOCKED"

    def test_consecutive_stages_counted(self):
        result = maturity_map(idea=True, design=True)
        assert result["highest_maturity"] == "DESIGN"


# ---------------------------------------------------------------------------
# proof_firewall
# ---------------------------------------------------------------------------

class TestProofFirewall:
    def test_empty_dims_blocked(self):
        result = proof_firewall({})
        assert result["verdict"] == "BLOCKED_FAIL_CLOSED"

    def test_missing_dim_blocked(self):
        result = proof_firewall({"DATA": 0.5, "RAW": 0.5, "SCORING": 0.5, "REPLAY": 0.5, "INDEPENDENCE": 0.5})
        assert result["verdict"] == "BLOCKED_FAIL_CLOSED"
        assert "SAFETY" in result["missing"]

    def test_safety_below_threshold_blocked(self):
        dims = {k: 0.8 for k in EVIDENCE_WEIGHTS}
        dims["SAFETY"] = 0.5
        result = proof_firewall(dims)
        assert result["verdict"] == "BLOCKED_FAIL_CLOSED"
        assert "SAFETY" in result["reason"]

    def test_all_dims_present_high_safety_ready(self):
        dims = {k: 0.9 for k in EVIDENCE_WEIGHTS}
        result = proof_firewall(dims)
        assert result["verdict"] == "READY_FOR_INDEPENDENT_REVIEW_NOT_PUBLIC_PROOF"
        assert result["missing"] == []

    def test_zero_dim_blocked(self):
        dims = {k: 0.9 for k in EVIDENCE_WEIGHTS}
        dims["RAW"] = 0.0
        result = proof_firewall(dims)
        assert result["verdict"] == "BLOCKED_FAIL_CLOSED"
        assert "RAW" in result["missing"]
