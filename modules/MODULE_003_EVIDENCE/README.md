# MODULE_003_EVIDENCE

Raw evidence validation and gating for the OmniAGIS audit pipeline.

## Function

Validates raw output records for hash integrity and completeness, audits ZIP artifacts,
loads JSONL files, validates metric passports, and computes numeric summaries of CSV files.
All operations are read-only — no writes to source data.

Key capabilities:
- **SHA-256 hashing**: Text, bytes, files, and JSON objects (canonical)
- **Raw record validation**: Checks required fields, hash integrity, non-empty output
- **Batch gate report**: Validates a set of records against an expected count; returns `pass_gate` bool
- **JSONL loading**: Skips blank/malformed lines; returns list of parsed dicts
- **Metric passport validation**: Checks required fields, hash chain (payload_hash, entry_hash)
- **ZIP audit**: Lists members, hashes contents, extracts run_summary and manifest metadata
- **CSV summary**: Numeric statistics (min, max, mean, count) for any CSV file

## Source

`gpts_core/evidence.py` — copied verbatim to `exports/evidence.py`

## Public API

| Symbol | Returns | Description |
|---|---|---|
| `sha256_text(text)` | `str` | SHA-256 of UTF-8 text, prefixed `sha256:` |
| `sha256_bytes(data)` | `str` | SHA-256 of raw bytes, prefixed `sha256:` |
| `sha256_file(path)` | `Optional[str]` | SHA-256 of file, or `None` if missing |
| `canonical_json(obj)` | `str` | Deterministic JSON (sorted keys) |
| `sha256_json(obj)` | `str` | SHA-256 of canonical JSON |
| `build_raw_record(...)` | `Dict` | Construct raw record with `output_hash` |
| `RecordVerdict` | dataclass | Single-record validation verdict |
| `validate_raw_record(record)` | `RecordVerdict` | Validate one raw record |
| `RawGateReport` | dataclass | Batch validation gate report |
| `validate_raw_records(...)` | `RawGateReport` | Validate a batch of records |
| `load_jsonl(path)` | `List[Dict]` | Load JSONL, skipping blank/malformed lines |
| `validate_metric_passport(obj, expect_cycle)` | `Dict` | Validate a metric passport dict |
| `audit_zip(path, max_hash_mb)` | `Dict` | Audit a ZIP artifact |
| `summarize_csv(path, max_cells)` | `Dict` | Numeric summary of a CSV file |

### `RecordVerdict` fields

```
output_id: str
valid: bool
errors: List[str]
computed_hash: Optional[str]
declared_hash: Optional[str]
to_dict() -> Dict
```

### `RawGateReport` fields

```
run_id: str
expected: int
present: int
empty: int
hash_valid: int
pass_gate: bool
scoring_status: str
verdict: str
records: List[Dict]
to_dict() -> Dict
```

## Dependencies

- `hashlib` (stdlib)
- `json` (stdlib)
- `zipfile` (stdlib)
- `csv` (stdlib)
- `pathlib` (stdlib)

## Coverage

66% (as measured against `tests/test_gpts_core.py::TestEvidence`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — all validation is local; no external audit or publication has been performed.

## Usage

```python
from modules.MODULE_003_EVIDENCE.exports.evidence import build_raw_record, validate_raw_records

rec = build_raw_record("run_01", "task_01", "out_01", "model_A", "raw output text")
report = validate_raw_records(run_id="run_01", expected=1, records=[rec])
print(report.pass_gate)   # True
print(report.verdict)     # PASS_RAW_COLLECTION_LOCAL
```

See `examples/example.py` for runnable examples.
