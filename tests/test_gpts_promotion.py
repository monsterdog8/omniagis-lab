"""Tests for gpts_core.promotion — promotion gate and seal/ledger audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpts_core.promotion import (
    _is_promotion_candidate,
    _row_blockers,
    evaluate_promotion,
    validate_seal_record,
    validate_seal_ledger,
    replay_jsonl_ledger,
)


# ---------------------------------------------------------------------------
# _is_promotion_candidate
# ---------------------------------------------------------------------------

def _passing_row():
    return {
        "provenance_chain": "COMPLETE",
        "dependency_resolution_status": "RESOLVED",
        "raw_ledger_presence": "PRESENT_JSONL",
        "replay_status": "REPLAY_PASS",
        "verdict": "VALIDATED",
        "claim_ceiling": "INTERNAL_ONLY",
    }


class TestIsPromotionCandidate:
    def test_passing_row_is_candidate(self):
        assert _is_promotion_candidate(_passing_row()) is True

    def test_quarantined_verdict_blocked(self):
        row = _passing_row()
        row["verdict"] = "QUARANTINED"
        assert _is_promotion_candidate(row) is False

    def test_local_only_ceiling_blocked(self):
        row = _passing_row()
        row["claim_ceiling"] = "LOCAL_ONLY"
        assert _is_promotion_candidate(row) is False

    def test_missing_required_signal_blocked(self):
        row = _passing_row()
        row["replay_status"] = "UNKNOWN"
        assert _is_promotion_candidate(row) is False

    def test_lab_only_ceiling_blocked(self):
        row = _passing_row()
        row["claim_ceiling"] = "LAB_ONLY"
        assert _is_promotion_candidate(row) is False

    def test_transcript_only_verdict_blocked(self):
        row = _passing_row()
        row["verdict"] = "TRANSCRIPT_ONLY"
        assert _is_promotion_candidate(row) is False


# ---------------------------------------------------------------------------
# evaluate_promotion
# ---------------------------------------------------------------------------

class TestEvaluatePromotion:
    def _write_matrix(self, path: Path, rows):
        import csv
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def test_no_candidates(self, tmp_path):
        p = tmp_path / "matrix.csv"
        row = _passing_row()
        row["claim_ceiling"] = "LOCAL_ONLY"
        self._write_matrix(p, [row])
        result = evaluate_promotion([p])
        assert result["global_verdict"] == "LOCKED_NO_VERIFIED_PROMOTION_CANDIDATE"
        assert result["candidates"] == []
        assert result["production_status"] == "LOCKED"

    def test_one_candidate(self, tmp_path):
        p = tmp_path / "matrix.csv"
        self._write_matrix(p, [_passing_row()])
        result = evaluate_promotion([p])
        assert result["global_verdict"] == "VERIFIED_CANDIDATES_PRESENT_REVIEW_REQUIRED"
        assert len(result["candidates"]) == 1

    def test_blocker_counted(self, tmp_path):
        p = tmp_path / "matrix.csv"
        row = _passing_row()
        row["replay_status"] = "UNKNOWN"
        self._write_matrix(p, [row])
        result = evaluate_promotion([p])
        assert result["blocker_count"] > 0

    def test_multiple_csv_files(self, tmp_path):
        p1 = tmp_path / "m1.csv"
        p2 = tmp_path / "m2.csv"
        self._write_matrix(p1, [_passing_row()])
        self._write_matrix(p2, [_passing_row()])
        result = evaluate_promotion([p1, p2])
        assert result["rows_evaluated"] == 2


# ---------------------------------------------------------------------------
# validate_seal_record
# ---------------------------------------------------------------------------

class TestValidateSealRecord:
    def _make_file(self, tmp_path, content=b"data"):
        f = tmp_path / "artifact.bin"
        f.write_bytes(content)
        return f

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _make_row(self, tmp_path, content=b"data"):
        f = self._make_file(tmp_path, content)
        sha = self._sha256(f)
        return {
            "schema": "seal_v1",
            "event_id": "001",
            "created_at_utc": "2024-01-01T00:00:00Z",
            "artifact_path": "artifact.bin",
            "artifact_exists": True,
            "expected_sha256": sha,
            "expected_bytes": f.stat().st_size,
            "recomputed_sha256": sha,
            "recomputed_bytes": f.stat().st_size,
            "verdict": "VALID",
            "proof_scope": "LOCAL",
            "blocked_claims": [],
            "claim_ceiling": "LOCAL_ONLY",
        }

    def test_valid_record_passes(self, tmp_path):
        row = self._make_row(tmp_path)
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is True

    def test_missing_field_fails(self, tmp_path):
        row = self._make_row(tmp_path)
        del row["schema"]
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is False
        assert any("missing:schema" in e for e in result["errors"])

    def test_sha256_mismatch_fails(self, tmp_path):
        row = self._make_row(tmp_path)
        row["expected_sha256"] = "wrong_sha"
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is False
        assert "sha256_mismatch" in result["errors"]

    def test_artifact_exists_mismatch_fails(self, tmp_path):
        row = self._make_row(tmp_path)
        row["artifact_exists"] = False
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is False
        assert "artifact_exists_mismatch" in result["errors"]


# ---------------------------------------------------------------------------
# validate_seal_ledger
# ---------------------------------------------------------------------------

class TestValidateSealLedger:
    def _make_seal_entry(self, tmp_path, artifact_name="a.bin", content=b"xyz"):
        f = tmp_path / artifact_name
        f.write_bytes(content)
        h = hashlib.sha256(content).hexdigest()
        return {
            "schema": "seal_v1",
            "event_id": artifact_name,
            "created_at_utc": "2024-01-01T00:00:00Z",
            "artifact_path": artifact_name,
            "artifact_exists": True,
            "expected_sha256": h,
            "expected_bytes": len(content),
            "recomputed_sha256": h,
            "recomputed_bytes": len(content),
            "verdict": "VALID",
            "proof_scope": "LOCAL",
            "blocked_claims": [],
            "claim_ceiling": "LOCAL_ONLY",
        }

    def test_all_valid_ledger(self, tmp_path):
        entries = [self._make_seal_entry(tmp_path, f"a{i}.bin", f"content{i}".encode()) for i in range(3)]
        ledger = tmp_path / "seal.jsonl"
        ledger.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        result = validate_seal_ledger(ledger, tmp_path)
        assert result["all_valid"] is True
        assert result["records"] == 3
        assert result["pass_count"] == 3

    def test_invalid_entry_counted(self, tmp_path):
        entry = self._make_seal_entry(tmp_path, "a0.bin", b"ok")
        entry["expected_sha256"] = "wrong"
        ledger = tmp_path / "seal.jsonl"
        ledger.write_text(json.dumps(entry), encoding="utf-8")
        result = validate_seal_ledger(ledger, tmp_path)
        assert result["all_valid"] is False
        assert result["fail_count"] == 1


# ---------------------------------------------------------------------------
# replay_jsonl_ledger
# ---------------------------------------------------------------------------

def _make_event(event_id, prev_hash=None, etype="test"):
    data = {}
    if prev_hash is not None:
        data["hash_parent"] = prev_hash
    return {"event_id": event_id, "timestamp": "2024-01-01T00:00:00Z",
            "type": etype, "data": data, "signature": "sig"}


def _hash_event(event):
    return hashlib.sha256(json.dumps(event, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class TestReplayJsonlLedger:
    def test_missing_file(self, tmp_path):
        result = replay_jsonl_ledger(tmp_path / "missing.jsonl")
        assert result["status"] == "FAIL"
        assert result["reason"] == "FILE_NOT_FOUND"

    def test_single_valid_event_no_chain(self, tmp_path):
        event = _make_event("000001")
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(event), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "PASS"
        assert result["events_count"] == 1

    def test_two_events_valid_chain(self, tmp_path):
        e1 = _make_event("000001")
        h1 = _hash_event(e1)
        e2 = _make_event("000002", prev_hash=h1)
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(e1) + "\n" + json.dumps(e2), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "PASS"
        assert result["events_count"] == 2

    def test_empty_line_fails(self, tmp_path):
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(_make_event("000001")) + "\n\n", encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "EMPTY_LINE"

    def test_non_monotonic_event_id_fails(self, tmp_path):
        e1 = _make_event("000002")
        e2 = _make_event("000001")
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(e1) + "\n" + json.dumps(e2), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "EVENT_ID_NOT_STRICTLY_INCREASING"

    def test_bad_parent_hash_fails(self, tmp_path):
        e1 = _make_event("000001")
        e2 = _make_event("000002", prev_hash="wrong_hash_here")
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(e1) + "\n" + json.dumps(e2), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "HASH_PARENT_MISMATCH"

    def test_genesis_with_zero_parent_ok(self, tmp_path):
        e1 = _make_event("000001", prev_hash="0" * 64)
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(e1), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "PASS"

    def test_genesis_with_wrong_parent_fails(self, tmp_path):
        e1 = _make_event("000001", prev_hash="wrong")
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(e1), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "GENESIS_PARENT_NOT_ZERO"

    def test_missing_required_field_fails(self, tmp_path):
        event = {"event_id": "000001", "timestamp": "2024-01-01T00:00:00Z", "type": "t"}
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(event), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "MISSING_FIELDS"

    def test_data_not_dict_fails(self, tmp_path):
        event = {"event_id": "000001", "timestamp": "x", "type": "t",
                 "data": "not_a_dict", "signature": "s"}
        f = tmp_path / "ledger.jsonl"
        f.write_text(json.dumps(event), encoding="utf-8")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "DATA_NOT_OBJECT"
