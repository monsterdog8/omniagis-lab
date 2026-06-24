"""Promotion gate engine and artifact continuity audit.

Reads EXOCHRONOS-style CSV matrices and evaluates rows for promotion readiness
using a strict set of required signal fields. Also validates artifact SHA-256
integrity from JSONL seal ledgers.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

_REQUIRED_SIGNALS: Dict[str, tuple] = {
    "provenance_chain": ("COMPLETE", "VERIFIED"),
    "dependency_resolution_status": ("RESOLVED", "NO_EXTERNAL_DEPENDENCY"),
    "raw_ledger_presence": ("PRESENT", "PRESENT_JSONL", "PRESENT_LOCAL"),
    "replay_status": ("PASS", "REPLAY_PASS", "BUNDLE_LEDGER_REPLAY_PASS",
                      "REPLAY_PASS_LOCAL_SYNTHETIC"),
}
_BLOCKING_VERDICTS = ("QUARANTINED", "LORE", "TRANSCRIPT_ONLY")
_BLOCKING_CEILINGS = ("LOCAL_ONLY", "LAB_ONLY", "DOCUMENTARY_ONLY",
                      "NOT_EXTERNAL_PROOF", "NOT_PRODUCTION")


def _token_match(value: str, tokens: Iterable[str]) -> bool:
    v = (value or "").upper()
    return any(tok in v for tok in tokens)


def _is_promotion_candidate(row: Dict[str, str]) -> bool:
    verdict = (row.get("verdict") or "").upper()
    ceiling = (row.get("claim_ceiling") or "").upper()
    if any(t in verdict for t in _BLOCKING_VERDICTS):
        return False
    if any(t in ceiling for t in _BLOCKING_CEILINGS):
        return False
    return all(_token_match(row.get(f, ""), tokens)
               for f, tokens in _REQUIRED_SIGNALS.items())


def _row_blockers(row: Dict[str, str]) -> List[str]:
    reasons = []
    if not _token_match(row.get("raw_ledger_presence", ""),
                        _REQUIRED_SIGNALS["raw_ledger_presence"]):
        reasons.append("RAW_LEDGER_MISSING_OR_NON_CANONICAL")
    if not _token_match(row.get("replay_status", ""),
                        _REQUIRED_SIGNALS["replay_status"]):
        reasons.append("REPLAY_NOT_PROMOTION_GRADE")
    if "QUARANTIN" in (row.get("verdict") or "").upper():
        reasons.append("QUARANTINED")
    return reasons


def read_matrix_csv(path: Path) -> List[Dict[str, str]]:
    """Read a promotion matrix CSV into a list of row dicts."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["_source"] = str(path)
            rows.append(row)
    return rows


def evaluate_promotion(paths: Iterable[Path]) -> Dict[str, Any]:
    """Evaluate promotion candidates across one or more CSV matrix files."""
    all_rows: List[Dict[str, str]] = []
    for p in paths:
        all_rows.extend(read_matrix_csv(p))

    candidates = [r for r in all_rows if _is_promotion_candidate(r)]
    blocker_rows = []
    for r in all_rows:
        reasons = _row_blockers(r)
        if reasons:
            blocker_rows.append({
                "artifact_id": r.get("artifact_id"),
                "filename": r.get("filename"),
                "source": r.get("_source"),
                "verdict": r.get("verdict"),
                "claim_ceiling": r.get("claim_ceiling"),
                "next_gate": r.get("next_gate"),
                "reasons": reasons,
            })

    verdict = (
        "VERIFIED_CANDIDATES_PRESENT_REVIEW_REQUIRED"
        if candidates else "LOCKED_NO_VERIFIED_PROMOTION_CANDIDATE"
    )
    return {
        "global_verdict": verdict,
        "production_status": "LOCKED",
        "rows_evaluated": len(all_rows),
        "candidates": candidates,
        "blocker_count": len(blocker_rows),
        "blocker_reason_counts": dict(
            Counter(r for b in blocker_rows for r in b["reasons"])
        ),
        "top_blockers": blocker_rows[:25],
    }


# ---------------------------------------------------------------------------
# Artifact continuity (seal audit)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_SEAL_REQUIRED = [
    "schema", "event_id", "created_at_utc", "artifact_path",
    "artifact_exists", "expected_sha256", "expected_bytes",
    "recomputed_sha256", "recomputed_bytes",
    "verdict", "proof_scope", "blocked_claims", "claim_ceiling",
]


def validate_seal_record(
    row: Dict[str, Any],
    root: Path,
) -> Dict[str, Any]:
    """Validate one SEAL continuity ledger record against the filesystem."""
    errors = [f"missing:{k}" for k in _SEAL_REQUIRED if k not in row]
    if errors:
        return {"event_id": row.get("event_id", "UNKNOWN"), "valid": False, "errors": errors}

    artifact = root / row["artifact_path"]
    exists = artifact.exists()
    actual_sha = _sha256_file(artifact) if exists else None
    actual_bytes = artifact.stat().st_size if exists else None

    if exists != row["artifact_exists"]:
        errors.append("artifact_exists_mismatch")
    if actual_sha != row.get("expected_sha256"):
        errors.append("sha256_mismatch")
    if actual_bytes != row.get("expected_bytes"):
        errors.append("bytes_mismatch")

    return {
        "event_id": row["event_id"],
        "artifact_path": row["artifact_path"],
        "valid": not errors,
        "errors": errors,
        "proof_scope": row.get("proof_scope"),
        "claim_ceiling": row.get("claim_ceiling"),
    }


def validate_seal_ledger(ledger_path: Path, root: Path) -> Dict[str, Any]:
    """Validate an entire SEAL continuity JSONL ledger."""
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    results = [validate_seal_record(r, root) for r in rows]
    pass_count = sum(1 for r in results if r["valid"])
    return {
        "ledger": str(ledger_path),
        "records": len(results),
        "pass_count": pass_count,
        "fail_count": len(results) - pass_count,
        "all_valid": pass_count == len(results),
        "results": results,
        "claim_ceiling": "LOCAL_SHA256_CONTINUITY_ONLY_NOT_SUBJECTIVE_PROOF",
    }


# ---------------------------------------------------------------------------
# JSONL ledger replay with hash-chain verification
# ---------------------------------------------------------------------------

def replay_jsonl_ledger(path: Path) -> Dict[str, Any]:
    """Replay a JSONL event ledger, verifying hash-parent chain and monotonic event_id."""
    _REQUIRED = {"event_id", "timestamp", "type", "data", "signature"}

    def _canonical(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _sha256(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    if not path.exists():
        return {"status": "FAIL", "reason": "FILE_NOT_FOUND", "path": str(path)}

    events = []
    prev_id: Optional[str] = None
    prev_hash: Optional[str] = None

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            return {"status": "FAIL", "reason": "EMPTY_LINE", "line": lineno}
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"status": "FAIL", "reason": "JSON_PARSE_ERROR", "line": lineno, "detail": str(exc)}

        missing = sorted(_REQUIRED - set(obj))
        if missing:
            return {"status": "FAIL", "reason": "MISSING_FIELDS", "line": lineno, "missing": missing}
        if not isinstance(obj.get("data"), dict):
            return {"status": "FAIL", "reason": "DATA_NOT_OBJECT", "line": lineno}

        eid = obj["event_id"]
        if prev_id is not None and eid <= prev_id:
            return {"status": "FAIL", "reason": "EVENT_ID_NOT_STRICTLY_INCREASING", "line": lineno}

        declared_parent = obj["data"].get("hash_parent")
        if eid == "000001":
            if declared_parent is not None and declared_parent != "0" * 64:
                return {"status": "FAIL", "reason": "GENESIS_PARENT_NOT_ZERO", "line": lineno}
        elif declared_parent is not None and prev_hash is not None:
            if declared_parent != prev_hash:
                return {"status": "FAIL", "reason": "HASH_PARENT_MISMATCH", "line": lineno}

        event_hash = _sha256(_canonical(obj))
        events.append({"event_id": eid, "type": obj["type"], "sha256": event_hash})
        prev_id = eid
        prev_hash = event_hash

    return {
        "status": "PASS",
        "path": str(path),
        "events_count": len(events),
        "latest_event_id": events[-1]["event_id"] if events else None,
        "claim_ceiling": "LOCAL_REPLAY_ONLY_NOT_EXTERNAL_PROOF",
    }
