"""Tests for MODULE_005_LEDGER.

Extracted from tests/test_gpts_core.py::TestLedger.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestLedger:
    def test_log_and_chain_valid(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        ledger.log("EVT_A", {"val": 1})
        ledger.log("EVT_B", {"val": 2})
        chain = ledger.verify_chain()
        assert chain["valid"] is True
        assert len(ledger.entries()) == 2

    def test_single_entry_valid(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        e = ledger.log("TEST", {"x": 42})
        assert any(k in e for k in ("sha256", "hash", "entry_hash", "event_hash"))
        chain = ledger.verify_chain()
        assert chain["valid"] is True

    def test_entries_returns_list(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        ledger.log("E1", {})
        ledger.log("E2", {})
        entries = ledger.entries()
        assert len(entries) == 2

    def test_empty_ledger_chain(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        chain = ledger.verify_chain()
        assert chain["valid"] is True
        assert len(ledger.entries()) == 0

    def test_citation_lock_has_citation(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock()
        corpus = ["the quick brown fox jumps over the lazy dog"]
        assert lock.has_citation(corpus, "quick brown fox")
        assert not lock.has_citation(corpus, "slow green turtle")

    def test_citation_lock_require_passes(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock()
        lock.require_citation(["hello world"], "hello world")

    def test_citation_lock_require_raises(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock()
        with pytest.raises(Exception):
            lock.require_citation(["hello world"], "NOT IN CORPUS")

    # Additional coverage tests

    def test_chained_ledger_verify_valid(self):
        from ledger import AuditLedger
        ledger = AuditLedger(chained=True)
        ledger.log("A", {"n": 1}, audit=True)
        ledger.log("B", {"n": 2}, audit=True)
        ledger.log("C", {"n": 3}, audit=True)
        result = ledger.verify_chain()
        assert result["valid"] is True
        assert result.get("entries_checked") == 3

    def test_chained_entry_has_prev_hash(self):
        from ledger import AuditLedger
        ledger = AuditLedger(chained=True)
        ledger.log("FIRST", {}, audit=True)
        e2 = ledger.log("SECOND", {}, audit=True)
        assert "prev_hash" in e2
        assert e2["prev_hash"].startswith("sha256:")

    def test_non_chained_verify_returns_valid(self):
        from ledger import AuditLedger
        ledger = AuditLedger(chained=False)
        ledger.log("X", {"data": "val"})
        result = ledger.verify_chain()
        assert result["valid"] is True
        assert "note" in result

    def test_file_persistence(self, tmp_path):
        from ledger import AuditLedger
        p = tmp_path / "ledger.jsonl"
        ledger = AuditLedger(path=p, chained=True)
        ledger.log("PERSIST_A", {"x": 1}, audit=True)
        ledger.log("PERSIST_B", {"x": 2}, audit=True)
        assert p.exists()

        # Reload and verify
        ledger2 = AuditLedger(path=p, chained=True)
        assert len(ledger2.entries()) == 2
        result = ledger2.verify_chain()
        assert result["valid"] is True

    def test_gate_blocks_when_not_ready(self):
        from ledger import AuditLedger, _SyncGate
        gate = _SyncGate(guard_stable=False)
        ledger = AuditLedger(gate=gate)
        with pytest.raises(RuntimeError, match="LEDGER_GATE_NOT_READY"):
            ledger.log("BLOCKED_EVENT", {})

    def test_gate_ready_when_all_true(self):
        from ledger import _SyncGate
        gate = _SyncGate(guard_stable=True, experts_synced=True,
                         rules_locked=True, context_frozen=True)
        assert gate.ready() is True

    def test_gate_not_ready_when_any_false(self):
        from ledger import _SyncGate
        for kwargs in [
            {"guard_stable": False},
            {"experts_synced": False},
            {"rules_locked": False},
            {"context_frozen": False},
        ]:
            gate = _SyncGate(**kwargs)
            assert gate.ready() is False

    def test_log_entry_has_timestamp(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        e = ledger.log("TIMED", {"v": 99})
        assert "timestamp" in e
        assert "Z" in e["timestamp"] or "+" in e["timestamp"]

    def test_log_entry_has_entry_hash(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        e = ledger.log("HASH_CHECK", {"v": 1})
        assert "entry_hash" in e
        assert e["entry_hash"].startswith("sha256:")

    def test_citation_lock_empty_value_returns_false(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock()
        corpus = ["some content here"]
        assert lock.has_citation(corpus, "") is False
        assert lock.has_citation(corpus, "   ") is False

    def test_citation_lock_none_value_raises(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_MISSING"):
            lock.require_citation(["any content"], None)

    def test_citation_lock_caches_result(self):
        from ledger import ContextCitationLock
        lock = ContextCitationLock(max_cache=10)
        corpus = ["alpha beta gamma"]
        # First lookup populates cache
        r1 = lock.has_citation(corpus, "beta")
        # Second lookup uses cache (corpus is exhausted as list but cache hit)
        r2 = lock.has_citation([""], "beta")
        assert r1 is True
        assert r2 is True

    def test_ledger_entries_are_copies(self):
        from ledger import AuditLedger
        ledger = AuditLedger()
        ledger.log("E", {"k": "v"})
        entries1 = ledger.entries()
        entries2 = ledger.entries()
        assert entries1 == entries2
        assert entries1 is not entries2
