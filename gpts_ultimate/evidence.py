"""Raw evidence validation and gating.

Validates raw output records (hash integrity, required fields), audits ZIP artifacts,
and summarizes CSV files numerically. All operations are read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_SHA256_PREFIX = "sha256:"


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return _SHA256_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return _SHA256_PREFIX + h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


# ---------------------------------------------------------------------------
# Raw record structure
# ---------------------------------------------------------------------------

_RAW_RECORD_FIELDS = [
    "run_id", "task_id", "output_id", "model_slot",
    "raw_output", "token_count", "latency_ms", "timestamp", "output_hash",
]


def build_raw_record(
    run_id: str,
    task_id: str,
    output_id: str,
    model_slot: str,
    raw_output: str,
    token_count: int = 0,
    latency_ms: int = 0,
    timestamp: str = "1970-01-01T00:00:00Z",
) -> Dict[str, Any]:
    """Construct a raw output record with a SHA-256 hash of the output text."""
    return {
        "run_id": run_id,
        "task_id": task_id,
        "output_id": output_id,
        "model_slot": model_slot,
        "raw_output": raw_output,
        "token_count": int(token_count),
        "latency_ms": int(latency_ms),
        "timestamp": timestamp,
        "output_hash": sha256_text(raw_output),
    }


@dataclass(frozen=True)
class RecordVerdict:
    output_id: str
    valid: bool
    errors: List[str]
    computed_hash: Optional[str]
    declared_hash: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_raw_record(record: Dict[str, Any]) -> RecordVerdict:
    """Check a single raw record for completeness and hash integrity."""
    output_id = str(record.get("output_id", "UNKNOWN"))
    errors: List[str] = []

    for field in _RAW_RECORD_FIELDS:
        if field not in record:
            errors.append(f"MISSING_FIELD:{field}")

    raw = record.get("raw_output", "")
    if not isinstance(raw, str):
        errors.append("RAW_OUTPUT_NOT_STRING")
        raw = ""
    if raw == "":
        errors.append("RAW_OUTPUT_EMPTY")

    declared = str(record.get("output_hash", "") or "")
    computed = sha256_text(raw)
    if not declared.startswith(_SHA256_PREFIX) or len(declared) != 71:
        errors.append("OUTPUT_HASH_FORMAT_INVALID")
    elif declared != computed:
        errors.append("OUTPUT_HASH_MISMATCH")

    for field in ("token_count", "latency_ms"):
        try:
            if int(record.get(field, -1)) < 0:
                errors.append(f"NEGATIVE_FIELD:{field}")
        except Exception:
            errors.append(f"NON_INTEGER_FIELD:{field}")

    return RecordVerdict(output_id, not errors, errors, computed, declared)


@dataclass(frozen=True)
class RawGateReport:
    run_id: str
    expected: int
    present: int
    empty: int
    hash_valid: int
    pass_gate: bool
    scoring_status: str
    verdict: str
    records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_raw_records(
    run_id: str,
    expected: int,
    records: Iterable[Dict[str, Any]],
    strict: bool = True,
    scoring_done: bool = False,
) -> RawGateReport:
    """Validate a batch of raw records against expected count."""
    recs = list(records)
    verdicts = [validate_raw_record(r) for r in recs]
    present = len(recs)
    empty = sum(1 for r in recs if r.get("raw_output", "") == "")
    hash_valid = sum(1 for v in verdicts if v.valid)
    pass_gate = bool(
        expected > 0 and present == expected and empty == 0
        and hash_valid == expected and strict
    )
    scoring_status = (
        "READY" if pass_gate and not scoring_done
        else ("DONE_LOCAL" if pass_gate and scoring_done else "BLOCKED")
    )
    return RawGateReport(
        run_id=run_id,
        expected=int(expected),
        present=present,
        empty=empty,
        hash_valid=hash_valid,
        pass_gate=pass_gate,
        scoring_status=scoring_status,
        verdict="PASS_RAW_COLLECTION_LOCAL" if pass_gate else "BLOCKED_FAIL_CLOSED",
        records=[v.to_dict() for v in verdicts],
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file, skipping blank or malformed lines."""
    result: List[Dict[str, Any]] = []
    if not path.exists():
        return result
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            result.append(obj if isinstance(obj, dict) else {"_line": i, "raw_output": "", "output_hash": ""})
        except json.JSONDecodeError:
            result.append({"_line": i, "raw_output": "", "output_hash": "", "_error": "PARSE_ERROR"})
    return result


# ---------------------------------------------------------------------------
# Metric passport validation
# ---------------------------------------------------------------------------

_PASSPORT_REQUIRED = [
    "metric_namespace", "formula_id", "source_backend", "record_semantics",
    "cycle_semantics", "raw_payload_status", "hash_status", "replay_status",
    "cycle", "timestamp_utc", "raw_payload", "payload_hash", "entry_hash",
]
_PASSPORT_CRITICAL = [
    "metric_namespace", "formula_id", "source_backend", "record_semantics",
    "cycle_semantics", "raw_payload_status", "hash_status", "replay_status",
]


def validate_metric_passport(
    obj: Dict[str, Any],
    expect_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate the structure and hash integrity of a metric passport."""
    if not isinstance(obj, dict):
        return {"valid": False, "blockers": ["NOT_AN_OBJECT"]}
    blockers: List[str] = []
    for f in _PASSPORT_REQUIRED:
        if f not in obj:
            blockers.append(f"MISSING:{f}")
    for f in _PASSPORT_CRITICAL:
        if obj.get(f) in (None, "", "UNKNOWN"):
            blockers.append(f"EMPTY_CRITICAL:{f}")
    if expect_cycle is not None and obj.get("cycle") != expect_cycle:
        blockers.append(f"CYCLE_MISMATCH_EXPECTED_{expect_cycle}")
    if isinstance(obj.get("raw_payload"), dict):
        if obj.get("payload_hash") != sha256_json(obj["raw_payload"]):
            blockers.append("PAYLOAD_HASH_MISMATCH")
        entry_base = {**obj, "entry_hash": _SHA256_PREFIX + "0" * 64}
        if obj.get("entry_hash") != sha256_json(entry_base):
            blockers.append("ENTRY_HASH_MISMATCH")
    else:
        blockers.append("RAW_PAYLOAD_NOT_OBJECT")
    return {"valid": not blockers, "blockers": blockers}


# ---------------------------------------------------------------------------
# ZIP audit
# ---------------------------------------------------------------------------

def audit_zip(path: Path, max_hash_mb: float = 64.0) -> Dict[str, Any]:
    """Audit a ZIP file: list members, extract SHA-256, look for run_summary/manifest."""
    if not path.exists():
        return {"exists": False, "path": str(path), "verdict": "NOT_FOUND"}
    members: List[Dict[str, Any]] = []
    max_bytes = int(max_hash_mb * 1024 * 1024)
    run_summary = None
    manifest = None
    with zipfile.ZipFile(path, "r") as z:
        for info in z.infolist():
            row: Dict[str, Any] = {
                "name": info.filename,
                "bytes": info.file_size,
                "sha256": None,
            }
            if info.file_size <= max_bytes and not info.is_dir():
                data = z.read(info.filename)
                row["sha256"] = sha256_bytes(data)
                if info.filename.endswith("run_summary.json"):
                    try:
                        run_summary = json.loads(data.decode("utf-8"))
                    except Exception:
                        pass
                if info.filename.endswith("manifest.json"):
                    try:
                        manifest = json.loads(data.decode("utf-8"))
                    except Exception:
                        pass
            members.append(row)
    replay = (run_summary or {}).get("ledger_replay", {})
    replay_pass = (
        replay.get("status") == "PASS"
        and int(replay.get("mismatch_count", 1)) == 0
    )
    return {
        "exists": True,
        "path": str(path),
        "zip_sha256": sha256_file(path),
        "member_count": len(members),
        "total_bytes": sum(int(m["bytes"]) for m in members),
        "members": members,
        "run_summary": run_summary,
        "manifest_meta": {k: (manifest or {}).get(k) for k in
                          ("schema", "version", "file_count", "aggregate_hash",
                           "public_proof", "production_status")} if manifest else None,
        "ledger_replay": replay,
        "replay_pass": replay_pass,
        "verdict": "REPLAY_PASS_LOCAL" if replay_pass else "REPLAY_NOT_ESTABLISHED",
        "proof_status": "LOCAL_ONLY_NOT_EXTERNAL_PROOF",
    }


# ---------------------------------------------------------------------------
# CSV numeric summary
# ---------------------------------------------------------------------------

def summarize_csv(path: Path, max_cells: int = 5_000_000) -> Dict[str, Any]:
    """Compute numeric summary statistics for a CSV file."""
    if not path.exists():
        return {"path": str(path), "exists": False, "verdict": "MISSING"}
    rows = cols_max = count = non_numeric = 0
    total = 0.0
    minv, maxv = math.inf, -math.inf
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            rows += 1
            cols_max = max(cols_max, len(row))
            for cell in row:
                if count >= max_cells:
                    continue
                try:
                    v = float(cell)
                    if math.isfinite(v):
                        total += v
                        count += 1
                        minv = min(minv, v)
                        maxv = max(maxv, v)
                except Exception:
                    non_numeric += 1
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "max_columns": cols_max,
        "numeric_cells": count,
        "non_numeric_cells": non_numeric,
        "min": None if count == 0 else minv,
        "max": None if count == 0 else maxv,
        "mean": (total / count) if count else None,
        "verdict": "NUMERIC_SUMMARY" if count else "NO_NUMERIC_CELLS",
    }
