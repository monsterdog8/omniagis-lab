"""Tests for gpts_core.evidence — raw record validation and gating."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from gpts_core.evidence import (
    sha256_text,
    sha256_bytes,
    sha256_file,
    canonical_json,
    sha256_json,
    build_raw_record,
    validate_raw_record,
    validate_raw_records,
    load_jsonl,
    validate_metric_passport,
    summarize_csv,
    audit_zip,
)


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

class TestSha256Helpers:
    def test_sha256_text_prefix(self):
        h = sha256_text("hello")
        assert h.startswith("sha256:")
        assert len(h) == 71

    def test_sha256_text_deterministic(self):
        assert sha256_text("abc") == sha256_text("abc")

    def test_sha256_bytes_prefix(self):
        h = sha256_bytes(b"data")
        assert h.startswith("sha256:")

    def test_sha256_file_existing(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"content")
        h = sha256_file(f)
        assert h is not None
        assert h.startswith("sha256:")

    def test_sha256_file_missing(self, tmp_path):
        assert sha256_file(tmp_path / "missing.txt") is None

    def test_canonical_json_sorted(self):
        obj = {"b": 2, "a": 1}
        s = canonical_json(obj)
        assert s == '{"a":1,"b":2}'

    def test_sha256_json_deterministic(self):
        obj = {"x": [1, 2, 3]}
        assert sha256_json(obj) == sha256_json(obj)


# ---------------------------------------------------------------------------
# build_raw_record
# ---------------------------------------------------------------------------

class TestBuildRawRecord:
    def test_output_hash_matches_raw_output(self):
        r = build_raw_record("run1", "t1", "o1", "gpt4", "hello world")
        expected = "sha256:" + hashlib.sha256("hello world".encode("utf-8")).hexdigest()
        assert r["output_hash"] == expected

    def test_all_required_fields_present(self):
        r = build_raw_record("r", "t", "o", "m", "text")
        for field in ["run_id", "task_id", "output_id", "model_slot",
                      "raw_output", "token_count", "latency_ms", "timestamp", "output_hash"]:
            assert field in r

    def test_token_count_is_int(self):
        r = build_raw_record("r", "t", "o", "m", "text", token_count=42)
        assert r["token_count"] == 42


# ---------------------------------------------------------------------------
# validate_raw_record
# ---------------------------------------------------------------------------

class TestValidateRawRecord:
    def _make_valid(self, text="hello"):
        return build_raw_record("run1", "task1", "out1", "gpt4", text, token_count=10, latency_ms=100)

    def test_valid_record_passes(self):
        r = self._make_valid()
        verdict = validate_raw_record(r)
        assert verdict.valid is True
        assert verdict.errors == []

    def test_missing_field_fails(self):
        r = self._make_valid()
        del r["run_id"]
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("MISSING_FIELD:run_id" in e for e in verdict.errors)

    def test_empty_raw_output_fails(self):
        r = self._make_valid()
        r["raw_output"] = ""
        r["output_hash"] = sha256_text("")
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("RAW_OUTPUT_EMPTY" in e for e in verdict.errors)

    def test_hash_mismatch_fails(self):
        r = self._make_valid()
        r["output_hash"] = sha256_text("wrong content")
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("OUTPUT_HASH_MISMATCH" in e for e in verdict.errors)

    def test_bad_hash_format_fails(self):
        r = self._make_valid()
        r["output_hash"] = "not-a-hash"
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("OUTPUT_HASH_FORMAT_INVALID" in e for e in verdict.errors)

    def test_negative_token_count_fails(self):
        r = self._make_valid()
        r["token_count"] = -1
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("NEGATIVE_FIELD:token_count" in e for e in verdict.errors)

    def test_non_integer_field_fails(self):
        r = self._make_valid()
        r["token_count"] = "not_an_int"
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("NON_INTEGER_FIELD:token_count" in e for e in verdict.errors)

    def test_non_string_raw_output_fails(self):
        r = self._make_valid()
        r["raw_output"] = 123
        verdict = validate_raw_record(r)
        assert verdict.valid is False
        assert any("RAW_OUTPUT_NOT_STRING" in e for e in verdict.errors)

    def test_to_dict_has_valid_key(self):
        r = self._make_valid()
        d = validate_raw_record(r).to_dict()
        assert "valid" in d


# ---------------------------------------------------------------------------
# validate_raw_records (batch)
# ---------------------------------------------------------------------------

class TestValidateRawRecords:
    def _make_record(self, idx=1, text="hello"):
        return build_raw_record(f"run1", f"t{idx}", f"o{idx}", "gpt4", text,
                                token_count=10, latency_ms=50)

    def test_all_valid_passes(self):
        records = [self._make_record(i) for i in range(1, 4)]
        report = validate_raw_records("run1", 3, records)
        assert report.pass_gate is True
        assert report.verdict == "PASS_RAW_COLLECTION_LOCAL"

    def test_fewer_records_than_expected_fails(self):
        records = [self._make_record(1)]
        report = validate_raw_records("run1", 3, records)
        assert report.pass_gate is False

    def test_empty_raw_output_fails(self):
        r = self._make_record(1)
        r["raw_output"] = ""
        report = validate_raw_records("run1", 1, [r])
        assert report.pass_gate is False
        assert report.empty == 1

    def test_scoring_status_ready(self):
        records = [self._make_record(i) for i in range(1, 3)]
        report = validate_raw_records("run1", 2, records, scoring_done=False)
        assert report.scoring_status == "READY"

    def test_scoring_status_done(self):
        records = [self._make_record(i) for i in range(1, 3)]
        report = validate_raw_records("run1", 2, records, scoring_done=True)
        assert report.scoring_status == "DONE_LOCAL"

    def test_to_dict_has_verdict(self):
        records = [self._make_record(1)]
        report = validate_raw_records("run1", 1, records)
        d = report.to_dict()
        assert "verdict" in d
        assert "pass_gate" in d


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_jsonl(tmp_path / "missing.jsonl")
        assert result == []

    def test_valid_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        result = load_jsonl(f)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
        result = load_jsonl(f)
        assert len(result) == 2

    def test_malformed_line_included_as_error(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\nNOT_JSON\n', encoding="utf-8")
        result = load_jsonl(f)
        assert len(result) == 2
        assert result[1].get("_error") == "PARSE_ERROR"

    def test_non_dict_line_wrapped(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('[1, 2, 3]\n', encoding="utf-8")
        result = load_jsonl(f)
        assert "_line" in result[0]


# ---------------------------------------------------------------------------
# validate_metric_passport
# ---------------------------------------------------------------------------

class TestValidateMetricPassport:
    def _make_passport(self):
        payload = {"metric": "value", "score": 0.5}
        payload_hash = sha256_json(payload)
        entry_base = {
            "metric_namespace": "ns.test",
            "formula_id": "f001",
            "source_backend": "local",
            "record_semantics": "computed",
            "cycle_semantics": "per_run",
            "raw_payload_status": "VALID",
            "hash_status": "VERIFIED",
            "replay_status": "PASS",
            "cycle": 1,
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "raw_payload": payload,
            "payload_hash": payload_hash,
            "entry_hash": "sha256:" + "0" * 64,
        }
        from gpts_core.evidence import sha256_json as sj
        entry_base["entry_hash"] = sj(entry_base)
        return entry_base

    def test_valid_passport_passes(self):
        passport = self._make_passport()
        result = validate_metric_passport(passport)
        assert result["valid"] is True
        assert result["blockers"] == []

    def test_not_dict_fails(self):
        result = validate_metric_passport("string")
        assert result["valid"] is False
        assert "NOT_AN_OBJECT" in result["blockers"]

    def test_missing_field_blocked(self):
        p = self._make_passport()
        del p["formula_id"]
        result = validate_metric_passport(p)
        assert result["valid"] is False
        assert any("MISSING:formula_id" in b for b in result["blockers"])

    def test_cycle_mismatch_blocked(self):
        p = self._make_passport()
        result = validate_metric_passport(p, expect_cycle=999)
        assert result["valid"] is False
        assert any("CYCLE_MISMATCH" in b for b in result["blockers"])


# ---------------------------------------------------------------------------
# summarize_csv
# ---------------------------------------------------------------------------

class TestSummarizeCsv:
    def test_numeric_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        result = summarize_csv(f)
        assert result["exists"] is True
        assert result["numeric_cells"] == 6
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(6.0)
        assert result["mean"] == pytest.approx(3.5)
        assert result["verdict"] == "NUMERIC_SUMMARY"

    def test_missing_file(self, tmp_path):
        result = summarize_csv(tmp_path / "missing.csv")
        assert result["exists"] is False
        assert result["verdict"] == "MISSING"

    def test_non_numeric_cells_counted(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\nhello,1.0\nworld,2.0\n", encoding="utf-8")
        result = summarize_csv(f)
        assert result["non_numeric_cells"] >= 2

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("", encoding="utf-8")
        result = summarize_csv(f)
        assert result["verdict"] == "NO_NUMERIC_CELLS"
        assert result["min"] is None

    def test_max_cells_limit(self, tmp_path):
        f = tmp_path / "big.csv"
        f.write_text("a\n" + "\n".join("1.0" for _ in range(20)), encoding="utf-8")
        result = summarize_csv(f, max_cells=5)
        assert result["numeric_cells"] == 5


# ---------------------------------------------------------------------------
# audit_zip
# ---------------------------------------------------------------------------

class TestAuditZip:
    def test_missing_zip(self, tmp_path):
        result = audit_zip(tmp_path / "missing.zip")
        assert result["exists"] is False
        assert result["verdict"] == "NOT_FOUND"

    def test_basic_zip(self, tmp_path):
        import zipfile as zf
        p = tmp_path / "test.zip"
        with zf.ZipFile(p, "w") as z:
            z.writestr("file.txt", "hello world")
        result = audit_zip(p)
        assert result["exists"] is True
        assert result["member_count"] == 1
        assert result["replay_pass"] is False
        assert result["verdict"] == "REPLAY_NOT_ESTABLISHED"

    def test_zip_with_run_summary_replay_pass(self, tmp_path):
        import json
        import zipfile as zf
        p = tmp_path / "test.zip"
        run_summary = {
            "ledger_replay": {"status": "PASS", "mismatch_count": 0}
        }
        with zf.ZipFile(p, "w") as z:
            z.writestr("run_summary.json", json.dumps(run_summary))
        result = audit_zip(p)
        assert result["replay_pass"] is True
        assert result["verdict"] == "REPLAY_PASS_LOCAL"

    def test_zip_sha256_present(self, tmp_path):
        import zipfile as zf
        p = tmp_path / "test.zip"
        with zf.ZipFile(p, "w") as z:
            z.writestr("f.txt", "data")
        result = audit_zip(p)
        assert result["zip_sha256"] is not None
        assert result["zip_sha256"].startswith("sha256:")
