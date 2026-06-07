"""Tests for MODULE_003_EVIDENCE.

Extracted from tests/test_gpts_core.py::TestEvidence.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestEvidence:
    def test_sha256_text_deterministic(self):
        from evidence import sha256_text
        h1 = sha256_text("hello")
        h2 = sha256_text("hello")
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_sha256_text_different_inputs(self):
        from evidence import sha256_text
        assert sha256_text("a") != sha256_text("b")

    def test_build_raw_record_fields(self):
        from evidence import build_raw_record
        rec = build_raw_record("run_01", "task_01", "out_01", "model_A", "raw text")
        assert rec["run_id"] == "run_01"
        assert rec["task_id"] == "task_01"
        assert any("hash" in k or "sha256" in k for k in rec)

    def test_validate_raw_record_valid(self):
        from evidence import build_raw_record, validate_raw_record
        rec = build_raw_record("run_01", "task_01", "out_01", "model_A", "raw text")
        v = validate_raw_record(rec)
        assert hasattr(v, "valid") or isinstance(v, dict)

    def test_validate_raw_records_pass(self):
        from evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("run_01", "task_01", "out_01", "model_A", "raw text")
        report = validate_raw_records(run_id="run_01", expected=1, records=[rec])
        assert hasattr(report, "pass_gate") or isinstance(report, dict)

    def test_validate_raw_records_fail_on_wrong_count(self):
        from evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("run_01", "task_01", "out_01", "model_A", "raw text")
        report = validate_raw_records(run_id="run_01", expected=5, records=[rec])
        if hasattr(report, "pass_gate"):
            assert not report.pass_gate
        else:
            assert not report.get("pass_gate", True)

    def test_sha256_file(self, tmp_path):
        from evidence import sha256_file
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        h = sha256_file(f)
        assert h is not None
        assert h.startswith("sha256:")

    def test_sha256_file_missing(self, tmp_path):
        from evidence import sha256_file
        h = sha256_file(tmp_path / "nonexistent.txt")
        assert h is None

    def test_audit_zip(self, tmp_path):
        import zipfile
        from evidence import audit_zip
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("file.txt", "content")
        result = audit_zip(zp)
        assert isinstance(result, dict)

    def test_summarize_csv(self, tmp_path):
        from evidence import summarize_csv
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        result = summarize_csv(f)
        assert isinstance(result, dict)
        assert result.get("row_count", 0) >= 0

    def test_load_jsonl(self, tmp_path):
        from evidence import load_jsonl
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        records = load_jsonl(f)
        assert len(records) == 2
        assert records[0]["a"] == 1

    # Additional coverage tests

    def test_sha256_bytes_deterministic(self):
        from evidence import sha256_bytes
        h1 = sha256_bytes(b"hello")
        h2 = sha256_bytes(b"hello")
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_sha256_bytes_different_inputs(self):
        from evidence import sha256_bytes
        assert sha256_bytes(b"x") != sha256_bytes(b"y")

    def test_canonical_json_sorted_keys(self):
        from evidence import canonical_json
        obj = {"z": 1, "a": 2, "m": 3}
        s = canonical_json(obj)
        assert '"a"' in s
        assert '"z"' in s
        assert s.index('"a"') < s.index('"z"')

    def test_sha256_json_deterministic(self):
        from evidence import sha256_json
        obj = {"key": "value", "n": 42}
        h1 = sha256_json(obj)
        h2 = sha256_json(obj)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_build_raw_record_hash_matches(self):
        from evidence import build_raw_record, sha256_text
        rec = build_raw_record("r", "t", "o", "m", "test output")
        expected_hash = sha256_text("test output")
        assert rec["output_hash"] == expected_hash

    def test_validate_raw_record_hash_mismatch(self):
        from evidence import build_raw_record, validate_raw_record
        rec = build_raw_record("r", "t", "o", "m", "original text")
        # Tamper with the output
        tampered = dict(rec)
        tampered["raw_output"] = "different text"
        v = validate_raw_record(tampered)
        assert not v.valid
        assert any("MISMATCH" in e or "HASH" in e for e in v.errors)

    def test_validate_raw_records_pass_gate_true(self):
        from evidence import build_raw_record, validate_raw_records
        recs = [
            build_raw_record("run_X", f"task_{i}", f"out_{i}", "model_A", f"output {i}")
            for i in range(3)
        ]
        report = validate_raw_records(run_id="run_X", expected=3, records=recs)
        assert report.pass_gate is True
        assert report.present == 3
        assert report.hash_valid == 3
        assert report.empty == 0

    def test_validate_raw_records_scoring_status_ready(self):
        from evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("r", "t", "o", "m", "out")
        report = validate_raw_records(run_id="r", expected=1, records=[rec], scoring_done=False)
        assert report.scoring_status == "READY"

    def test_validate_raw_records_scoring_status_done_local(self):
        from evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("r", "t", "o", "m", "out")
        report = validate_raw_records(run_id="r", expected=1, records=[rec], scoring_done=True)
        assert report.scoring_status == "DONE_LOCAL"

    def test_load_jsonl_missing_file(self, tmp_path):
        from evidence import load_jsonl
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_load_jsonl_malformed_line(self, tmp_path):
        from evidence import load_jsonl
        f = tmp_path / "bad.jsonl"
        f.write_text('{"a": 1}\nNOT JSON\n{"b": 2}\n')
        records = load_jsonl(f)
        # Should have 3 entries (malformed gets error record)
        assert len(records) == 3

    def test_audit_zip_missing_file(self, tmp_path):
        from evidence import audit_zip
        result = audit_zip(tmp_path / "missing.zip")
        assert result["exists"] is False
        assert result["verdict"] == "NOT_FOUND"

    def test_summarize_csv_missing_file(self, tmp_path):
        from evidence import summarize_csv
        result = summarize_csv(tmp_path / "missing.csv")
        assert result["exists"] is False

    def test_summarize_csv_numeric_stats(self, tmp_path):
        from evidence import summarize_csv
        f = tmp_path / "nums.csv"
        f.write_text("x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        result = summarize_csv(f)
        assert result["numeric_cells"] > 0
        assert result["min"] == 1.0
        assert result["max"] == 6.0

    def test_validate_metric_passport_missing_fields(self):
        from evidence import validate_metric_passport
        result = validate_metric_passport({})
        assert result["valid"] is False
        assert len(result["blockers"]) > 0

    def test_validate_metric_passport_not_dict(self):
        from evidence import validate_metric_passport
        result = validate_metric_passport("not a dict")
        assert result["valid"] is False

    def test_record_verdict_to_dict(self):
        from evidence import build_raw_record, validate_raw_record
        rec = build_raw_record("r", "t", "o", "m", "output text")
        v = validate_raw_record(rec)
        d = v.to_dict()
        assert isinstance(d, dict)
        assert "valid" in d
        assert "output_id" in d

    def test_raw_gate_report_to_dict(self):
        from evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("r", "t", "o", "m", "text")
        report = validate_raw_records(run_id="r", expected=1, records=[rec])
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "pass_gate" in d
