"""Append-only audit ledger with SHA-256 chain integrity.

Two modes:
  - hot (fast): hash event JSON, return hash + timestamp.
  - cold (chained): chain-hash by appending prev_hash to event, maintain chain.

Also includes LongContextLock for proof-of-citation from a streaming corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _SyncGate:
    """Lightweight pre-condition checker before ledger accepts events."""

    def __init__(
        self,
        guard_stable: bool = True,
        experts_synced: bool = True,
        rules_locked: bool = True,
        context_frozen: bool = True,
    ) -> None:
        self._ok = guard_stable and experts_synced and rules_locked and context_frozen

    def ready(self) -> bool:
        return self._ok


class AuditLedger:
    """Thread-safe append-only ledger.

    Each entry is hashed; in chained mode each entry also includes the previous hash,
    creating a tamper-evident chain.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        chained: bool = False,
        gate: Optional[_SyncGate] = None,
    ) -> None:
        self._path = path
        self._chained = chained
        self._gate = gate or _SyncGate()
        self._chain_last: str = "sha256:" + "0" * 64
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                self._entries.append(json.loads(line))
            except Exception:
                pass
        if self._entries and self._chained:
            self._chain_last = self._entries[-1].get("entry_hash", self._chain_last)

    def _append_file(self, entry: Dict[str, Any]) -> None:
        if self._path:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    def log(self, event: str, data: Any = None, *, audit: bool = False) -> Dict[str, Any]:
        """Append an event. If audit=True, uses chained mode."""
        if not self._gate.ready():
            raise RuntimeError("LEDGER_GATE_NOT_READY")
        payload = {"event": event, "data": data, "timestamp": _utc_now()}
        with self._lock:
            if audit and self._chained:
                enriched = {**payload, "prev_hash": self._chain_last}
                entry_hash = _sha256(_canonical(enriched))
                entry = {**enriched, "entry_hash": entry_hash}
                self._chain_last = entry_hash
            else:
                entry_hash = _sha256(_canonical(payload))
                entry = {**payload, "entry_hash": entry_hash}
            self._entries.append(entry)
            self._append_file(entry)
        return entry

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def verify_chain(self) -> Dict[str, Any]:
        """Verify chain integrity. Returns dict with valid:bool and first_error."""
        with self._lock:
            entries = list(self._entries)
        if not self._chained:
            return {"valid": True, "note": "Non-chained ledger — no chain to verify."}
        prev = "sha256:" + "0" * 64
        for i, e in enumerate(entries):
            if e.get("prev_hash") != prev:
                return {"valid": False, "first_error": i, "reason": "PREV_HASH_MISMATCH"}
            declared = e.get("entry_hash", "")
            base = {k: v for k, v in e.items() if k != "entry_hash"}
            computed = _sha256(_canonical(base))
            if declared != computed:
                return {"valid": False, "first_error": i, "reason": "ENTRY_HASH_MISMATCH"}
            prev = declared
        return {"valid": True, "entries_checked": len(entries)}


# ---------------------------------------------------------------------------
# Citation-proof lock for long-context corpora
# ---------------------------------------------------------------------------

class ContextCitationLock:
    """Enforce proof-of-citation: a claimed value must appear verbatim in a corpus.

    Useful for preventing hallucinated facts in long-context settings.
    """

    def __init__(self, max_cache: int = 1024) -> None:
        self._cache: Dict[str, bool] = {}
        self._max_cache = max_cache

    def _cache_put(self, key: str, value: bool) -> None:
        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def has_citation(
        self,
        corpus: Iterable[str],
        quoted_value: str,
        exact: bool = True,
    ) -> bool:
        """Return True if quoted_value is found verbatim in the corpus stream."""
        if not quoted_value or not quoted_value.strip():
            return False
        key = "sha256:" + hashlib.sha256(quoted_value.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        pattern = re.escape(quoted_value) if exact else quoted_value
        for chunk in corpus:
            if re.search(pattern, chunk):
                self._cache_put(key, True)
                return True
        self._cache_put(key, False)
        return False

    def require_citation(
        self,
        corpus: Iterable[str],
        claimed_value: Optional[str],
        exact: bool = True,
    ) -> None:
        """Raise ValueError if claimed_value cannot be found in corpus (fail-closed)."""
        if claimed_value is None:
            raise ValueError("CITATION_MISSING")
        if not self.has_citation(corpus, claimed_value, exact=exact):
            raise ValueError("CITATION_NOT_FOUND_IN_CORPUS")
