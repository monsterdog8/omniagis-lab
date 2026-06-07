# MODULE_009_PROMOTION

## Purpose

Promotion gate engine and artifact continuity audit.

Reads EXOCHRONOS-style CSV matrices and evaluates rows for promotion readiness using a strict set of required signal fields (provenance chain, dependency resolution, raw ledger presence, replay status). Also validates artifact SHA-256 integrity from JSONL seal ledgers and replays JSONL event ledgers with hash-parent chain verification.

## Source

`gpts_core/promotion.py`

## Public API

| Function | Signature | Returns | Description |
|---|---|---|---|
| `read_matrix_csv` | `(path: Path) -> List[Dict[str, str]]` | `List[Dict[str, str]]` | Read promotion matrix CSV |
| `evaluate_promotion` | `(paths: Iterable[Path]) -> Dict[str, Any]` | `Dict[str, Any]` | Evaluate promotion candidates across CSV files |
| `validate_seal_record` | `(row: Dict, root: Path) -> Dict[str, Any]` | `Dict[str, Any]` | Validate one SEAL continuity record |
| `validate_seal_ledger` | `(ledger_path: Path, root: Path) -> Dict[str, Any]` | `Dict[str, Any]` | Validate full SEAL JSONL ledger |
| `replay_jsonl_ledger` | `(path: Path) -> Dict[str, Any]` | `Dict[str, Any]` | Replay JSONL event ledger with hash-chain check |

## Promotion Signal Requirements

A row is a promotion candidate only when ALL of the following are satisfied:

- `provenance_chain`: `COMPLETE` or `VERIFIED`
- `dependency_resolution_status`: `RESOLVED` or `NO_EXTERNAL_DEPENDENCY`
- `raw_ledger_presence`: `PRESENT`, `PRESENT_JSONL`, or `PRESENT_LOCAL`
- `replay_status`: `PASS`, `REPLAY_PASS`, `BUNDLE_LEDGER_REPLAY_PASS`, or `REPLAY_PASS_LOCAL_SYNTHETIC`

Additionally, rows with verdicts `QUARANTINED`, `LORE`, or `TRANSCRIPT_ONLY`, or claim ceilings containing `LOCAL_ONLY`, `LAB_ONLY`, `DOCUMENTARY_ONLY`, `NOT_EXTERNAL_PROOF`, or `NOT_PRODUCTION` are blocked.

## JSONL Ledger Replay

Required fields per event: `event_id`, `timestamp`, `type`, `data` (object), `signature`.

If `data` contains `hash_parent`, the hash-parent chain is verified. Event IDs must be strictly increasing.

## Dependencies

All stdlib: `csv`, `json`, `pathlib`, `hashlib`

## Coverage

65%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All outputs reflect local filesystem and in-memory ledger checks. No external validation is claimed.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_009.py` — extracted from `tests/test_gpts_core.py::TestPromotion`
