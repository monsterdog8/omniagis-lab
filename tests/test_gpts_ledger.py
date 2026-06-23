"""Tests for gpts_core.ledger — AuditLedger and ContextCitationLock."""
from __future__ import annotations

import json
import threading

import pytest

from gpts_core.ledger import AuditLedger, ContextCitationLock, _SyncGate


# ---------------------------------------------------------------------------
# _SyncGate
# ---------------------------------------------------------------------------

class TestSyncGate:
    def test_all_true_is_ready(self):
        g = _SyncGate(True, True, True, True)
        assert g.ready() is True

    def test_any_false_not_ready(self):
        g = _SyncGate(False, True, True, True)
        assert g.ready() is False

    def test_defaults_are_ready(self):
        g = _SyncGate()
        assert g.ready() is True


# ---------------------------------------------------------------------------
# AuditLedger — hot mode (no chaining)
# ---------------------------------------------------------------------------

class TestAuditLedgerHotMode:
    def test_log_returns_entry(self):
        ledger = AuditLedger()
        entry = ledger.log("test_event", {"k": "v"})
        assert entry["event"] == "test_event"
        assert entry["data"] == {"k": "v"}
        assert "entry_hash" in entry
        assert entry["entry_hash"].startswith("sha256:")

    def test_entries_accumulate(self):
        ledger = AuditLedger()
        ledger.log("e1")
        ledger.log("e2")
        assert len(ledger.entries()) == 2

    def test_entries_returns_copy(self):
        ledger = AuditLedger()
        ledger.log("e1")
        e = ledger.entries()
        e.clear()
        assert len(ledger.entries()) == 1

    def test_gate_not_ready_raises(self):
        gate = _SyncGate(guard_stable=False)
        ledger = AuditLedger(gate=gate)
        with pytest.raises(RuntimeError, match="LEDGER_GATE_NOT_READY"):
            ledger.log("event")

    def test_hot_mode_no_prev_hash_in_entry(self):
        ledger = AuditLedger()
        entry = ledger.log("e", audit=False)
        assert "prev_hash" not in entry

    def test_persists_to_file(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        ledger = AuditLedger(path=p)
        ledger.log("persist_test")
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["event"] == "persist_test"

    def test_loads_existing_file(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        ledger1 = AuditLedger(path=p)
        ledger1.log("first")
        ledger2 = AuditLedger(path=p)
        assert len(ledger2.entries()) == 1
        assert ledger2.entries()[0]["event"] == "first"


# ---------------------------------------------------------------------------
# AuditLedger — chained mode
# ---------------------------------------------------------------------------

class TestAuditLedgerChainedMode:
    def test_chained_entry_has_prev_hash(self):
        ledger = AuditLedger(chained=True)
        entry = ledger.log("e1", audit=True)
        assert "prev_hash" in entry
        assert entry["prev_hash"].startswith("sha256:")

    def test_chain_starts_at_zero_hash(self):
        ledger = AuditLedger(chained=True)
        entry = ledger.log("e1", audit=True)
        assert entry["prev_hash"] == "sha256:" + "0" * 64

    def test_second_entry_prev_hash_matches_first_entry_hash(self):
        ledger = AuditLedger(chained=True)
        e1 = ledger.log("first", audit=True)
        e2 = ledger.log("second", audit=True)
        assert e2["prev_hash"] == e1["entry_hash"]

    def test_verify_chain_valid(self):
        ledger = AuditLedger(chained=True)
        for i in range(5):
            ledger.log(f"event_{i}", audit=True)
        result = ledger.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 5

    def test_verify_chain_non_chained_is_valid(self):
        ledger = AuditLedger(chained=False)
        ledger.log("e1")
        result = ledger.verify_chain()
        assert result["valid"] is True
        assert "Non-chained" in result["note"]

    def test_verify_chain_detects_tampering(self):
        ledger = AuditLedger(chained=True)
        ledger.log("e1", audit=True)
        ledger.log("e2", audit=True)
        ledger._entries[0]["entry_hash"] = "sha256:" + "a" * 64
        result = ledger.verify_chain()
        assert result["valid"] is False

    def test_thread_safety(self):
        ledger = AuditLedger(chained=True)
        errors = []

        def writer():
            try:
                for i in range(20):
                    ledger.log(f"event_{i}", audit=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(ledger.entries()) == 80
        result = ledger.verify_chain()
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# ContextCitationLock
# ---------------------------------------------------------------------------

class TestContextCitationLock:
    def test_found_in_corpus(self):
        lock = ContextCitationLock()
        corpus = ["The quick brown fox", "jumps over the lazy dog"]
        assert lock.has_citation(corpus, "quick brown fox") is True

    def test_not_found_in_corpus(self):
        lock = ContextCitationLock()
        corpus = ["hello world"]
        assert lock.has_citation(corpus, "xyz abc") is False

    def test_empty_value_returns_false(self):
        lock = ContextCitationLock()
        assert lock.has_citation(["corpus text"], "") is False

    def test_whitespace_only_returns_false(self):
        lock = ContextCitationLock()
        assert lock.has_citation(["corpus text"], "   ") is False

    def test_caching_works(self):
        lock = ContextCitationLock()
        corpus = ["hello world"]
        result1 = lock.has_citation(corpus, "hello")
        result2 = lock.has_citation(["different corpus"], "hello")
        assert result1 is True
        assert result2 is True

    def test_cache_eviction(self):
        lock = ContextCitationLock(max_cache=2)
        corpus = ["a b c"]
        lock.has_citation(corpus, "a")
        lock.has_citation(corpus, "b")
        lock.has_citation(corpus, "c")
        assert len(lock._cache) == 2

    def test_require_citation_passes_when_found(self):
        lock = ContextCitationLock()
        lock.require_citation(["The value is 42"], "42")

    def test_require_citation_raises_when_not_found(self):
        lock = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_NOT_FOUND_IN_CORPUS"):
            lock.require_citation(["corpus"], "missing value")

    def test_require_citation_raises_on_none(self):
        lock = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_MISSING"):
            lock.require_citation(["corpus"], None)

    def test_inexact_match(self):
        lock = ContextCitationLock()
        corpus = ["The number is 42.0"]
        assert lock.has_citation(corpus, r"\d+\.\d+", exact=False) is True
