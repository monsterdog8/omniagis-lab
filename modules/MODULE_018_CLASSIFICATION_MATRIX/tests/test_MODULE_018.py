"""Tests for MODULE_018_CLASSIFICATION_MATRIX.

Loads classification_matrix_schema_v1.json from the exports directory and validates
the schema structure: auditors list, label_schema, c08_gate threshold, and required keys.
"""
from __future__ import annotations

import json
import pathlib

import pytest

_EXPORTS_DIR = pathlib.Path(__file__).parent.parent / "exports"
_SCHEMA_FILE = _EXPORTS_DIR / "classification_matrix_schema_v1.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    with _SCHEMA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestClassificationMatrixFile:
    def test_file_exists(self):
        assert _SCHEMA_FILE.exists(), f"Schema file not found: {_SCHEMA_FILE}"

    def test_file_is_valid_json(self):
        with _SCHEMA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)


class TestTopLevelKeys:
    def test_schema_key(self, schema):
        assert schema["_schema"] == "CLASSIFICATION_MATRIX"

    def test_version_key(self, schema):
        assert isinstance(schema["_version"], str)

    def test_claim_ceiling(self, schema):
        assert schema["_claim_ceiling"] == "LOCAL_ONLY__NO_EXTERNAL_PROOF"

    def test_status_is_template(self, schema):
        assert "TEMPLATE" in schema["_status"]

    def test_required_top_level_keys(self, schema):
        required = {
            "_schema", "_version", "_claim_ceiling", "_status",
            "packet_reference", "auditors", "label_schema",
            "classifications", "computed_metrics", "c08_gate", "pipeline_log",
        }
        missing = required - set(schema.keys())
        assert not missing, f"Missing top-level keys: {missing}"


class TestPacketReference:
    def test_packet_id_is_string(self, schema):
        pkt = schema["packet_reference"]
        assert isinstance(pkt["packet_id"], str)
        assert pkt["packet_id"] == "C01-C08"

    def test_packet_hash_is_64_hex(self, schema):
        h = schema["packet_reference"]["packet_hash"]
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h.lower())

    def test_canonical_is_bool(self, schema):
        assert isinstance(schema["packet_reference"]["canonical"], bool)
        assert schema["packet_reference"]["canonical"] is False


class TestAuditors:
    def test_auditors_is_list(self, schema):
        assert isinstance(schema["auditors"], list)

    def test_exactly_five_auditors(self, schema):
        assert len(schema["auditors"]) == 5, \
            f"Expected 5 auditors, got {len(schema['auditors'])}"

    def test_auditor_ids(self, schema):
        ids = {a["auditor_id"] for a in schema["auditors"]}
        expected = {"HUMAN_01", "HUMAN_02", "GPT", "CLAUDE", "GROK"}
        assert ids == expected, f"Auditor IDs mismatch: got {ids}"

    def test_auditor_types(self, schema):
        for a in schema["auditors"]:
            assert a["auditor_type"] in ("HUMAN", "AI"), \
                f"Unexpected auditor_type for {a['auditor_id']}: {a['auditor_type']}"

    def test_all_auditors_are_blind(self, schema):
        for a in schema["auditors"]:
            assert a["blind"] is True, \
                f"Auditor {a['auditor_id']} has blind={a['blind']!r}, expected True"

    def test_human_auditors_are_human(self, schema):
        human_auditors = {a["auditor_id"] for a in schema["auditors"] if a["auditor_type"] == "HUMAN"}
        assert "HUMAN_01" in human_auditors
        assert "HUMAN_02" in human_auditors

    def test_ai_auditors_are_ai(self, schema):
        ai_auditors = {a["auditor_id"] for a in schema["auditors"] if a["auditor_type"] == "AI"}
        assert "GPT" in ai_auditors
        assert "CLAUDE" in ai_auditors
        assert "GROK" in ai_auditors

    def test_auditors_have_required_fields(self, schema):
        for a in schema["auditors"]:
            for field in ("auditor_id", "auditor_type", "blind"):
                assert field in a, f"Auditor {a.get('auditor_id')!r} missing field: {field}"


class TestLabelSchema:
    def test_label_values_present(self, schema):
        ls = schema["label_schema"]
        assert "values" in ls
        assert isinstance(ls["values"], list)

    def test_exactly_four_labels(self, schema):
        labels = schema["label_schema"]["values"]
        assert len(labels) == 4, f"Expected 4 label values, got {len(labels)}: {labels}"

    def test_label_values_correct(self, schema):
        labels = set(schema["label_schema"]["values"])
        expected = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
        assert labels == expected, f"Label values mismatch: {labels}"

    def test_confidence_range_is_0_to_10(self, schema):
        cr = schema["label_schema"]["confidence_range"]
        assert isinstance(cr, list)
        assert len(cr) == 2
        assert cr[0] == 0
        assert cr[1] == 10

    def test_required_fields_is_list(self, schema):
        rf = schema["label_schema"]["required_fields"]
        assert isinstance(rf, list)
        assert len(rf) >= 1

    def test_required_fields_contains_key_items(self, schema):
        rf = set(schema["label_schema"]["required_fields"])
        for field in ("auditor_id", "claim_id", "label", "confidence"):
            assert field in rf, f"required_fields missing: {field}"


class TestClassifications:
    def test_classifications_is_list(self, schema):
        assert isinstance(schema["classifications"], list)

    def test_classification_entry_structure(self, schema):
        """Template entry should contain the documented field names."""
        entry = schema["classifications"][0]
        expected_fields = {"auditor_id", "claim_id", "label", "confidence", "rationale_brief", "timestamp"}
        for field in expected_fields:
            assert field in entry, f"Classification entry missing field: {field}"

    def test_template_entry_values_are_null(self, schema):
        """Template entries have null values awaiting real data."""
        entry = schema["classifications"][0]
        # At least some values should be null in the template
        null_fields = [k for k, v in entry.items() if v is None and k != "_note"]
        assert len(null_fields) >= 1, "Template entry should have null values"


class TestComputedMetrics:
    def test_computed_metrics_present(self, schema):
        assert "computed_metrics" in schema

    def test_computed_metrics_status_not_yet_computed(self, schema):
        cm = schema["computed_metrics"]
        assert cm["_status"] == "NOT_YET_COMPUTED"

    def test_krippendorff_alpha_null_in_template(self, schema):
        assert schema["computed_metrics"]["krippendorff_alpha"] is None

    def test_c08_gate_result_null_in_template(self, schema):
        assert schema["computed_metrics"]["c08_gate_result"] is None


class TestC08Gate:
    def test_c08_gate_present(self, schema):
        assert "c08_gate" in schema

    def test_alpha_threshold_minimum(self, schema):
        gate = schema["c08_gate"]
        assert "alpha_threshold_minimum" in gate
        threshold = gate["alpha_threshold_minimum"]
        assert isinstance(threshold, (int, float))
        assert abs(threshold - 0.667) < 1e-9, \
            f"alpha_threshold_minimum expected 0.667, got {threshold}"

    def test_alpha_threshold_substantial(self, schema):
        gate = schema["c08_gate"]
        assert "alpha_threshold_substantial" in gate
        threshold = gate["alpha_threshold_substantial"]
        assert isinstance(threshold, (int, float))
        assert abs(threshold - 0.800) < 1e-9, \
            f"alpha_threshold_substantial expected 0.800, got {threshold}"

    def test_minimum_threshold_less_than_substantial(self, schema):
        gate = schema["c08_gate"]
        assert gate["alpha_threshold_minimum"] < gate["alpha_threshold_substantial"]

    def test_blocked_claims_on_fail_is_list(self, schema):
        bcf = schema["c08_gate"]["blocked_claims_on_fail"]
        assert isinstance(bcf, list)
        assert len(bcf) >= 1
        assert all(isinstance(c, str) for c in bcf)

    def test_blocked_claims_include_external_replication(self, schema):
        bcf = schema["c08_gate"]["blocked_claims_on_fail"]
        assert "external_replication" in bcf

    def test_gate_result_null_in_template(self, schema):
        assert schema["c08_gate"]["result"] is None


class TestPipelineLog:
    def test_pipeline_log_present(self, schema):
        assert "pipeline_log" in schema
        assert isinstance(schema["pipeline_log"], dict)

    def test_pipeline_log_keys_present(self, schema):
        pl = schema["pipeline_log"]
        expected_keys = {
            "packet_confirmed", "auditors_assigned", "classifications_complete",
            "matrix_sealed", "alpha_computed", "c08_gate_evaluated", "meta_audit_complete",
        }
        missing = expected_keys - set(pl.keys())
        assert not missing, f"Missing pipeline_log keys: {missing}"

    def test_pipeline_log_all_null_in_template(self, schema):
        pl = schema["pipeline_log"]
        for k, v in pl.items():
            assert v is None, f"pipeline_log[{k!r}] should be null in template, got {v!r}"


class TestC08GateLogic:
    """Test the gate logic using the threshold values from the schema."""

    def test_alpha_below_minimum_fails(self, schema):
        threshold = schema["c08_gate"]["alpha_threshold_minimum"]
        alpha_simulated = 0.500
        assert alpha_simulated < threshold, "Alpha 0.500 should be below minimum threshold"

    def test_alpha_at_minimum_passes(self, schema):
        threshold = schema["c08_gate"]["alpha_threshold_minimum"]
        alpha_simulated = 0.667
        assert alpha_simulated >= threshold, "Alpha 0.667 should meet minimum threshold"

    def test_alpha_at_substantial_passes_substantial(self, schema):
        threshold = schema["c08_gate"]["alpha_threshold_substantial"]
        alpha_simulated = 0.850
        assert alpha_simulated >= threshold, "Alpha 0.850 should meet substantial threshold"

    def test_alpha_between_thresholds_passes_minimum_only(self, schema):
        min_t = schema["c08_gate"]["alpha_threshold_minimum"]
        sub_t = schema["c08_gate"]["alpha_threshold_substantial"]
        alpha_simulated = 0.720
        assert alpha_simulated >= min_t, "Alpha 0.720 meets minimum"
        assert alpha_simulated < sub_t, "Alpha 0.720 does not meet substantial"
