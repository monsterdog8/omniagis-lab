"""MODULE_009_PROMOTION — runnable usage examples.

Demonstrates: read_matrix_csv, evaluate_promotion, validate_seal_ledger,
and replay_jsonl_ledger.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

from gpts_core.promotion import (
    read_matrix_csv,
    evaluate_promotion,
    validate_seal_ledger,
    replay_jsonl_ledger,
)

# ---------------------------------------------------------------------------
# Example 1: Evaluate promotion candidates from a CSV matrix
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    # Write a promotion matrix CSV with one blocked row and one candidate row
    matrix_csv = tmp_path / "matrix.csv"
    matrix_csv.write_text(
        "artifact_id,filename,verdict,claim_ceiling,provenance_chain,"
        "dependency_resolution_status,raw_ledger_presence,replay_status,next_gate\n"
        "a001,blocked.py,LORE,LOCAL_ONLY,INCOMPLETE,UNRESOLVED,MISSING,FAIL,NONE\n"
        "a002,candidate.py,VERIFIED,BOUNDED,COMPLETE,RESOLVED,PRESENT,PASS,REVIEW\n"
    )

    rows = read_matrix_csv(matrix_csv)
    print(f"[Example 1] read_matrix_csv: {len(rows)} rows")

    result = evaluate_promotion([matrix_csv])
    print(f"[Example 1] global_verdict: {result['global_verdict']}")
    print(f"[Example 1] candidates: {len(result['candidates'])}")
    print(f"[Example 1] blocker_count: {result['blocker_count']}")
    print(f"[Example 1] production_status: {result['production_status']}")

# ---------------------------------------------------------------------------
# Example 2: Replay a valid JSONL event ledger
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    # Create a minimal valid JSONL ledger with one genesis event
    events = [
        {
            "event_id": "000001",
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "GENESIS",
            "data": {},
            "signature": "sig_genesis",
        },
        {
            "event_id": "000002",
            "timestamp": "2026-01-01T00:01:00Z",
            "type": "CAPTURE",
            "data": {"value": 0.82},
            "signature": "sig_capture",
        },
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    replay = replay_jsonl_ledger(ledger)
    print(f"\n[Example 2] replay_jsonl_ledger status: {replay['status']}")
    print(f"[Example 2] events_count: {replay['events_count']}")
    print(f"[Example 2] latest_event_id: {replay['latest_event_id']}")
    print(f"[Example 2] claim_ceiling: {replay['claim_ceiling']}")

# ---------------------------------------------------------------------------
# Example 3: Validate an empty SEAL ledger (zero records)
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    seal_ledger = tmp_path / "seal.jsonl"
    seal_ledger.write_text("")  # empty — zero records

    seal_result = validate_seal_ledger(seal_ledger, tmp_path)
    print(f"\n[Example 3] validate_seal_ledger records: {seal_result['records']}")
    print(f"[Example 3] all_valid: {seal_result['all_valid']}")
    print(f"[Example 3] claim_ceiling: {seal_result['claim_ceiling']}")
