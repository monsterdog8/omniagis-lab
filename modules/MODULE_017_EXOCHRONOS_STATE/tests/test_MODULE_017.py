"""Tests for MODULE_017_EXOCHRONOS_STATE.

Loads EXOCHRONOS_STATE_v1.json from the exports directory and validates
that all required keys are present and values are of the expected types.
"""
from __future__ import annotations

import json
import pathlib

import pytest

_EXPORTS_DIR = pathlib.Path(__file__).parent.parent / "exports"
_STATE_FILE = _EXPORTS_DIR / "EXOCHRONOS_STATE_v1.json"


@pytest.fixture(scope="module")
def state() -> dict:
    with _STATE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestExochronosStateSchema:
    def test_file_exists(self):
        assert _STATE_FILE.exists(), f"State file not found: {_STATE_FILE}"

    def test_top_level_schema_key(self, state):
        assert state["_schema"] == "EXOCHRONOS_STATE"

    def test_top_level_version(self, state):
        assert isinstance(state["_version"], str)
        assert state["_version"] == "1.0"

    def test_top_level_date(self, state):
        assert isinstance(state["_date"], str)
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", state["_date"]), \
            f"_date is not YYYY-MM-DD: {state['_date']}"

    def test_top_level_claim_ceiling(self, state):
        assert state["_claim_ceiling"] == "LOCAL_ONLY__NO_EXTERNAL_PROOF"

    def test_required_top_level_keys(self, state):
        required = {
            "_schema", "_version", "_date", "_claim_ceiling",
            "convergence", "component_status", "gap_assessment",
            "repo_metrics", "packet", "owner_decision_point",
            "next_phase", "axioms", "blocked_claims", "allowed_claims",
        }
        missing = required - set(state.keys())
        assert not missing, f"Missing top-level keys: {missing}"


class TestExochronosConvergence:
    def test_convergence_verdict_is_string(self, state):
        c = state["convergence"]
        assert isinstance(c["verdict"], str)
        assert c["verdict"] == "CONVERGENCE_DETECTED"

    def test_convergence_independent_audits_bool(self, state):
        assert isinstance(state["convergence"]["independent_audits_agree_on_bottleneck"], bool)
        assert state["convergence"]["independent_audits_agree_on_bottleneck"] is True

    def test_convergence_on_bottleneck_is_string(self, state):
        val = state["convergence"]["convergence_on_bottleneck"]
        assert isinstance(val, str)
        assert val == "SUPPORTED"

    def test_convergence_basis_is_list(self, state):
        basis = state["convergence"]["basis"]
        assert isinstance(basis, list)
        assert len(basis) >= 1
        assert all(isinstance(b, str) for b in basis)


class TestExochronosComponentStatus:
    def test_all_required_components_present(self, state):
        required = {
            "core_deterministic_framework",
            "governance_framework",
            "promotion_engine",
            "krippendorff_engine",
            "blindness_verifier",
            "replication_infrastructure",
            "external_replication",
            "real_auditor_classifications",
        }
        cs = state["component_status"]
        missing = required - set(cs.keys())
        assert not missing, f"Missing component_status keys: {missing}"

    def test_component_values_are_strings(self, state):
        for k, v in state["component_status"].items():
            assert isinstance(v, str), f"component_status[{k!r}] is not str: {v!r}"

    def test_ready_components_are_ready(self, state):
        cs = state["component_status"]
        ready = [
            "core_deterministic_framework",
            "governance_framework",
            "promotion_engine",
            "krippendorff_engine",
            "blindness_verifier",
            "replication_infrastructure",
        ]
        for component in ready:
            assert cs[component] == "READY", f"{component} expected READY, got {cs[component]!r}"

    def test_external_replication_not_started(self, state):
        assert state["component_status"]["external_replication"] == "NOT_STARTED"

    def test_real_auditor_classifications_missing(self, state):
        assert state["component_status"]["real_auditor_classifications"] == "MISSING"


class TestExochronosGapAssessment:
    def test_technical_gap_low(self, state):
        assert state["gap_assessment"]["technical_gap"] == "LOW"

    def test_data_gap_high(self, state):
        assert state["gap_assessment"]["data_gap"] == "HIGH"

    def test_gap_values_are_strings(self, state):
        for k, v in state["gap_assessment"].items():
            assert isinstance(v, str)


class TestExochronosRepoMetrics:
    def test_coverage_omniagis_is_string(self, state):
        assert isinstance(state["repo_metrics"]["coverage_omniagis"], str)
        assert "99" in state["repo_metrics"]["coverage_omniagis"]

    def test_coverage_gpts_core_is_string(self, state):
        assert isinstance(state["repo_metrics"]["coverage_gpts_core"], str)

    def test_equivalence_tests_is_int(self, state):
        assert isinstance(state["repo_metrics"]["equivalence_tests"], int)
        assert state["repo_metrics"]["equivalence_tests"] >= 1

    def test_equivalence_all_pass(self, state):
        rm = state["repo_metrics"]
        assert rm["equivalence_pass"] == rm["equivalence_tests"], \
            f"equivalence_pass={rm['equivalence_pass']} != equivalence_tests={rm['equivalence_tests']}"

    def test_unexplained_deltas_zero(self, state):
        assert state["repo_metrics"]["unexplained_deltas"] == 0


class TestExochronosPacket:
    def test_packet_id_is_string(self, state):
        assert isinstance(state["packet"]["id"], str)
        assert state["packet"]["id"] == "C01-C08"

    def test_packet_hash_is_64_hex(self, state):
        h = state["packet"]["packet_hash"]
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h.lower())

    def test_packet_frozen_true(self, state):
        assert state["packet"]["frozen"] is True

    def test_packet_canonical_false(self, state):
        assert state["packet"]["canonical"] is False

    def test_packet_canonical_status_is_string(self, state):
        assert isinstance(state["packet"]["canonical_status"], str)
        assert "AWAITING" in state["packet"]["canonical_status"]


class TestExochronosOwnerDecision:
    def test_option_a_is_string(self, state):
        assert isinstance(state["owner_decision_point"]["option_a"], str)

    def test_option_b_is_string(self, state):
        assert isinstance(state["owner_decision_point"]["option_b"], str)

    def test_rational_default_mentions_option_a(self, state):
        rd = state["owner_decision_point"]["rational_default"]
        assert isinstance(rd, str)
        assert "OPTION_A" in rd


class TestExochronosNextPhase:
    def test_gate_is_string(self, state):
        assert isinstance(state["next_phase"]["gate"], str)
        assert state["next_phase"]["gate"] == "OWNER_PACKET_DECISION"

    def test_pipeline_is_list(self, state):
        pipeline = state["next_phase"]["pipeline"]
        assert isinstance(pipeline, list)
        assert len(pipeline) >= 3
        assert all(isinstance(step, str) for step in pipeline)

    def test_blocked_until_real_classifications(self, state):
        assert "real_auditor_classifications" in state["next_phase"]["blocked_until"]


class TestExochronosAxioms:
    def test_all_three_axioms_present(self, state):
        axioms = state["axioms"]
        for ax in ("AX-01", "AX-02", "AX-03"):
            assert ax in axioms, f"Missing axiom: {ax}"

    def test_axiom_values_are_strings(self, state):
        for ax, val in state["axioms"].items():
            assert isinstance(val, str)


class TestExochronosClaimLists:
    def test_blocked_claims_is_non_empty_list(self, state):
        bc = state["blocked_claims"]
        assert isinstance(bc, list)
        assert len(bc) >= 1
        assert all(isinstance(c, str) for c in bc)

    def test_allowed_claims_is_non_empty_list(self, state):
        ac = state["allowed_claims"]
        assert isinstance(ac, list)
        assert len(ac) >= 1
        assert all(isinstance(c, str) for c in ac)

    def test_production_ready_is_blocked(self, state):
        blocked = state["blocked_claims"]
        assert any("production" in c.lower() for c in blocked), \
            "Expected 'production_ready' type claim in blocked_claims"
