# MODULE_005_LEDGER

Append-only SHA-256 chain audit log with citation locking for the OmniAGIS audit pipeline.

## Function

Provides two classes:

1. **`AuditLedger`**: Thread-safe append-only event log. Two modes:
   - *Hot mode* (`chained=False`): Each entry is independently SHA-256 hashed.
   - *Cold/chained mode* (`chained=True`): Each entry includes the previous entry's hash,
     creating a tamper-evident chain. `verify_chain()` checks integrity of the full chain.
   - Optional file persistence: pass a `Path` to persist entries as JSONL.
   - Gate guard: a `_SyncGate` instance must return `ready()=True` before any log call.

2. **`ContextCitationLock`**: Proof-of-citation enforcer for long-context settings.
   - `has_citation(corpus, value)`: streams a corpus and returns True if the value
     appears verbatim (or by regex if `exact=False`).
   - `require_citation(corpus, value)`: raises `ValueError` if citation not found.
   - Internal LRU-style cache keyed by SHA-256 of the quoted value.

## Source

`gpts_core/ledger.py` — copied verbatim to `exports/ledger.py`

## Public API

### `_SyncGate`

```python
_SyncGate(
    guard_stable: bool = True,
    experts_synced: bool = True,
    rules_locked: bool = True,
    context_frozen: bool = True,
)
ready() -> bool   # True only if all four flags were True at construction
```

### `AuditLedger`

```python
AuditLedger(
    path: Optional[Path] = None,
    chained: bool = False,
    gate: Optional[_SyncGate] = None,
)
log(event: str, data: Any = None, *, audit: bool = False) -> Dict
entries() -> List[Dict]
verify_chain() -> Dict     # {"valid": bool, ...}
```

### `ContextCitationLock`

```python
ContextCitationLock(max_cache: int = 1024)
has_citation(corpus: Iterable[str], quoted_value: str, exact: bool = True) -> bool
require_citation(corpus: Iterable[str], claimed_value: Optional[str], exact: bool = True) -> None
```

## Dependencies

- `hashlib` (stdlib)
- `json` (stdlib)
- `threading` (stdlib)
- `pathlib` (stdlib)
- `re` (stdlib)

## Coverage

64% (as measured against `tests/test_gpts_core.py::TestLedger`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — audit chain is locally verifiable only; no external certification has been performed.

## Usage

```python
from modules.MODULE_005_LEDGER.exports.ledger import AuditLedger, ContextCitationLock

# In-memory chained ledger
ledger = AuditLedger(chained=True)
ledger.log("RUN_START", {"run_id": "r001"}, audit=True)
ledger.log("SCORE_DONE", {"score": 0.88}, audit=True)
chain = ledger.verify_chain()
print(chain["valid"])   # True

# Citation lock
lock = ContextCitationLock()
corpus = ["The measured accuracy is 0.88 on cycle C07."]
lock.require_citation(corpus, "accuracy is 0.88")  # passes
```

See `examples/example.py` for runnable examples.
