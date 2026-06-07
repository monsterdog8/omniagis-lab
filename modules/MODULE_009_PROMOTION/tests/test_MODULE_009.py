"""Tests for MODULE_009_PROMOTION — extracted from tests/test_gpts_core.py::TestPromotion."""
from __future__ import annotations

import hashlib
import json

import pytest

from gpts_core.promotion import (
    replay_jsonl_ledger,
    validate_seal_ledger,
    evaluate_promotion,
)


class TestPromotion:
    def test_replay_jsonl_missing_file(self, tmp_path):
        result = replay_jsonl_ledger(tmp_path / "nonexistent.jsonl")
        assert result["status"] == "FAIL"

    def test_replay_jsonl_valid_chain(self, tmp_path):
        events = [
            {"event_id": "000001", "timestamp": "2026-01-01T00:00:00Z",
             "type": "GENESIS", "data": {}, "signature": "sig1"},
        ]
        f = tmp_path / "ledger.jsonl"
        f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "PASS"
        assert result["events_count"] == 1

    def test_replay_jsonl_empty_line_fails(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"event_id":"000001","timestamp":"T","type":"X","data":{},"signature":"s"}\n\n')
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"

    def test_replay_jsonl_missing_fields(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"event_id": "000001"}\n')
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "MISSING_FIELDS"

    def test_validate_seal_ledger_empty(self, tmp_path):
        f = tmp_path / "seal.jsonl"
        f.write_text("")
        result = validate_seal_ledger(f, tmp_path)
        assert result["records"] == 0
        assert result["all_valid"] is True

    def test_evaluate_promotion_no_candidates(self, tmp_path):
        csv = tmp_path / "matrix.csv"
        csv.write_text(
            "artifact_id,filename,verdict,claim_ceiling,provenance_chain,"
            "dependency_resolution_status,raw_ledger_presence,replay_status,next_gate\n"
            "a001,test.py,LORE,LOCAL_ONLY,INCOMPLETE,UNRESOLVED,MISSING,FAIL,NONE\n"
        )
        result = evaluate_promotion([csv])
        assert result["global_verdict"] == "LOCKED_NO_VERIFIED_PROMOTION_CANDIDATE"
        assert result["candidates"] == []

    def test_evaluate_promotion_candidate(self, tmp_path):
        csv = tmp_path / "matrix.csv"
        csv.write_text(
            "artifact_id,filename,verdict,claim_ceiling,provenance_chain,"
            "dependency_resolution_status,raw_ledger_presence,replay_status,next_gate\n"
            "a001,ok.py,VERIFIED,BOUNDED,COMPLETE,RESOLVED,PRESENT,PASS,REVIEW\n"
        )
        result = evaluate_promotion([csv])
        assert len(result["candidates"]) == 1
