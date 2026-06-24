"""Tests for gpts_ultimate.gate — claim classification, evidence scoring, scientific bridges."""
import pytest

from gpts_ultimate.gate import (
    ClaimStatus,
    EvidenceInput,
    classify_claim,
    classify_claim_scientific,
    compute_evidence_score,
    from_gate_counts,
    maturity_map,
    proof_firewall,
    validate_metric,
)


# ---------------------------------------------------------------------------
# classify_claim
# ---------------------------------------------------------------------------
class TestClassifyClaim:
    def test_blocked_production_ready(self):
        r = classify_claim("This model is production-ready and approved for production")
        assert r.status == ClaimStatus.BLOCKED.value

    def test_blocked_consciousness(self):
        r = classify_claim("consciousness proven in lab")
        assert r.status == ClaimStatus.BLOCKED.value

    def test_blocked_global_superiority(self):
        r = classify_claim("beats GPT on every benchmark")
        assert r.status == ClaimStatus.BLOCKED.value

    def test_allowed_bounded_simulation(self):
        r = classify_claim("this is a local simulation prototype hypothesis")
        assert r.status == ClaimStatus.ALLOWED_BOUNDED.value

    def test_unknown_neutral(self):
        r = classify_claim("the value was 42 in the experiment")
        assert r.status == ClaimStatus.UNKNOWN.value

    def test_empty_string(self):
        r = classify_claim("")
        assert r.status in (ClaimStatus.UNKNOWN.value, ClaimStatus.ALLOWED_BOUNDED.value)

    def test_none_treated_as_empty(self):
        r = classify_claim(None)
        assert r.status in (ClaimStatus.UNKNOWN.value, ClaimStatus.ALLOWED_BOUNDED.value)

    def test_hits_list_populated(self):
        r = classify_claim("production-ready system")
        assert len(r.hits) > 0


# ---------------------------------------------------------------------------
# compute_evidence_score
# ---------------------------------------------------------------------------
class TestComputeEvidenceScore:
    def test_all_zeros_blocked(self):
        inp = EvidenceInput()
        score = compute_evidence_score(inp)
        assert score.final_score == 0.0
        assert not score.public_claim_allowed

    def test_full_evidence_passes(self):
        inp = EvidenceInput(
            DATA=1.0, RAW=1.0, SCORING=1.0, REPLAY=1.0,
            INDEPENDENCE=1.0, SAFETY=1.0,
            raw_expected=10, raw_valid=10,
            scoring_done=True, replay_available=True,
            independent_review=True,
        )
        score = compute_evidence_score(inp)
        assert score.public_claim_allowed

    def test_penalties_reduce_score(self):
        inp_full = EvidenceInput(
            DATA=1.0, RAW=1.0, SCORING=1.0, REPLAY=1.0,
            INDEPENDENCE=1.0, SAFETY=1.0,
            raw_expected=10, raw_valid=10,
            scoring_done=True, replay_available=True, independent_review=True,
        )
        inp_partial = EvidenceInput(
            DATA=1.0, RAW=0.5, SCORING=0.0, REPLAY=0.0,
            INDEPENDENCE=0.0, SAFETY=0.5,
            raw_expected=10, raw_valid=5,
        )
        assert compute_evidence_score(inp_full).final_score > compute_evidence_score(inp_partial).final_score

    def test_scoring_status_blocked_when_no_raw(self):
        inp = EvidenceInput(raw_expected=10, raw_valid=0)
        score = compute_evidence_score(inp)
        assert score.scoring_status == "BLOCKED"


# ---------------------------------------------------------------------------
# from_gate_counts
# ---------------------------------------------------------------------------
class TestFromGateCounts:
    def test_zero_expected(self):
        inp = from_gate_counts(expected=0, present=0, valid=0)
        assert inp.DATA == 0.0

    def test_full_counts(self):
        inp = from_gate_counts(expected=10, present=10, valid=10, safety=0.9)
        assert inp.RAW == 1.0
        assert inp.DATA == 1.0

    def test_partial_valid(self):
        inp = from_gate_counts(expected=10, present=10, valid=5)
        assert inp.RAW == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# validate_metric (bridge to ValidationPipeline)
# ---------------------------------------------------------------------------
class TestValidateMetric:
    def test_pass_high_r2(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", 0.5, r_squared=0.97)
        assert result["status"] == "PASS"

    def test_conditional(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", 0.5, r_squared=0.85)
        assert result["status"] == "CONDITIONAL"

    def test_fail_low_r2(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", 0.5, r_squared=0.50)
        assert result["status"] == "FAIL"

    def test_unknown_no_r2(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", 0.5)
        assert result["status"] == "UNKNOWN"

    def test_invalid_nan(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", float("nan"))
        assert result["status"] == "INVALID"

    def test_invalid_inf(self):
        result = validate_metric("lyapunov", "LAMBDA_1_LOGISTIC", float("inf"))
        assert result["status"] == "INVALID"


# ---------------------------------------------------------------------------
# classify_claim_scientific
# ---------------------------------------------------------------------------
class TestClassifyClaimScientific:
    def test_none_observation_unknown(self):
        assert classify_claim_scientific(None) == "UNKNOWN"

    def test_high_r2_observed(self):
        assert classify_claim_scientific("obs", evidence_r2=0.96) == "OBSERVED"

    def test_medium_r2_with_theory_supported(self):
        assert classify_claim_scientific("obs", evidence_r2=0.85, theory_support=True) == "SUPPORTED"

    def test_plausible_r2(self):
        assert classify_claim_scientific("obs", evidence_r2=0.70) == "PLAUSIBLE"

    def test_negative_r2_refuted(self):
        assert classify_claim_scientific("obs", evidence_r2=-0.1) == "REFUTED"

    def test_no_r2_unknown(self):
        assert classify_claim_scientific("obs") == "UNKNOWN"

    def test_r2_between_80_no_theory(self):
        result = classify_claim_scientific("obs", evidence_r2=0.85, theory_support=False)
        assert result == "PLAUSIBLE"


# ---------------------------------------------------------------------------
# maturity_map
# ---------------------------------------------------------------------------
class TestMaturityMap:
    def test_public_claim_always_blocked(self):
        m = maturity_map(idea=True, design=True, prototype=True)
        assert m["public_claim_right"] == "BLOCKED"

    def test_highest_maturity_tracks(self):
        m = maturity_map(idea=True, design=True)
        assert m["highest_maturity"] == "DESIGN"

    def test_none_completed(self):
        m = maturity_map()
        assert m["highest_maturity"] == "NONE"


# ---------------------------------------------------------------------------
# proof_firewall
# ---------------------------------------------------------------------------
class TestProofFirewall:
    def test_empty_dims_blocked(self):
        r = proof_firewall({})
        assert r["verdict"] == "BLOCKED_FAIL_CLOSED"

    def test_full_dims_ready_for_review(self):
        dims = {"DATA": 1.0, "RAW": 1.0, "SCORING": 1.0,
                "REPLAY": 1.0, "INDEPENDENCE": 1.0, "SAFETY": 0.8}
        r = proof_firewall(dims)
        assert r["verdict"] == "READY_FOR_INDEPENDENT_REVIEW_NOT_PUBLIC_PROOF"

    def test_low_safety_blocked(self):
        dims = {"DATA": 1.0, "RAW": 1.0, "SCORING": 1.0,
                "REPLAY": 1.0, "INDEPENDENCE": 1.0, "SAFETY": 0.5}
        r = proof_firewall(dims)
        assert r["verdict"] == "BLOCKED_FAIL_CLOSED"
