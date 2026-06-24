"""Tests for gpts_ultimate.ledger — AuditLedger, ContextCitationLock, RealityLedger, HazardRegistry."""
import pytest
from pathlib import Path

from gpts_ultimate.ledger import (
    AuditLedger,
    ContextCitationLock,
    HazardRegistry,
    HazardStatus,
    Prediction,
    RealityLedger,
)


# ---------------------------------------------------------------------------
# AuditLedger
# ---------------------------------------------------------------------------
class TestAuditLedger:
    def test_log_returns_entry(self):
        ledger = AuditLedger()
        entry = ledger.log("TEST_EVENT", data={"x": 1})
        assert entry["event"] == "TEST_EVENT"
        assert "entry_hash" in entry

    def test_entries_accumulated(self):
        ledger = AuditLedger()
        ledger.log("A")
        ledger.log("B")
        assert len(ledger.entries()) == 2

    def test_non_chained_verify(self):
        ledger = AuditLedger(chained=False)
        ledger.log("E1")
        result = ledger.verify_chain()
        assert result["valid"] is True
        assert "Non-chained" in result.get("note", "")

    def test_chained_verify_passes(self):
        ledger = AuditLedger(chained=True)
        ledger.log("E1", audit=True)
        ledger.log("E2", audit=True)
        result = ledger.verify_chain()
        assert result["valid"] is True

    def test_chained_verify_detects_tampering(self):
        ledger = AuditLedger(chained=True)
        ledger.log("E1", audit=True)
        ledger.log("E2", audit=True)
        # tamper: change entry_hash of first entry
        ledger._entries[0]["entry_hash"] = "sha256:" + "f" * 64
        result = ledger.verify_chain()
        assert result["valid"] is False

    def test_persist_to_file(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = AuditLedger(path=path, chained=True)
        ledger.log("PERSIST_TEST", audit=True)
        assert path.exists()
        # reload
        ledger2 = AuditLedger(path=path, chained=True)
        assert len(ledger2.entries()) == 1

    def test_gate_not_ready_raises(self):
        from gpts_ultimate.ledger import _SyncGate
        gate = _SyncGate(guard_stable=False)
        ledger = AuditLedger(gate=gate)
        with pytest.raises(RuntimeError, match="LEDGER_GATE_NOT_READY"):
            ledger.log("FAIL")

    def test_thread_safety(self):
        import threading
        ledger = AuditLedger()
        threads = [threading.Thread(target=lambda: ledger.log("T")) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(ledger.entries()) == 20


# ---------------------------------------------------------------------------
# ContextCitationLock
# ---------------------------------------------------------------------------
class TestContextCitationLock:
    def test_found_in_corpus(self):
        ccl = ContextCitationLock()
        assert ccl.has_citation(["hello world", "foo bar"], "hello world") is True

    def test_not_found(self):
        ccl = ContextCitationLock()
        assert ccl.has_citation(["alpha beta"], "gamma") is False

    def test_empty_value_false(self):
        ccl = ContextCitationLock()
        assert ccl.has_citation(["anything"], "") is False

    def test_require_citation_raises_when_missing(self):
        ccl = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_NOT_FOUND"):
            ccl.require_citation(["alpha"], "beta")

    def test_require_citation_raises_none(self):
        ccl = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_MISSING"):
            ccl.require_citation(["alpha"], None)

    def test_cache_used(self):
        ccl = ContextCitationLock()
        corpus = ["hello world"]
        ccl.has_citation(corpus, "hello world")
        # second call hits cache — no exception
        assert ccl.has_citation(corpus, "hello world") is True


# ---------------------------------------------------------------------------
# RealityLedger (co-imported from ledger via math_metrics)
# ---------------------------------------------------------------------------
class TestRealityLedger:
    def test_list_all(self):
        preds = RealityLedger.list_all()
        assert "P_001" in preds
        assert "P_002" in preds

    def test_get_prediction(self):
        p = RealityLedger.get_prediction("P_001")
        assert isinstance(p, Prediction)
        assert p.status == "FROZEN"

    def test_get_nonexistent(self):
        assert RealityLedger.get_prediction("P_999") is None

    def test_prediction_fields(self):
        p = RealityLedger.get_prediction("P_002")
        assert p.confidence == pytest.approx(0.92)
        assert p.difficulty == "LOW"
        assert p.outcome is None


# ---------------------------------------------------------------------------
# HazardRegistry (co-imported from ledger via math_metrics)
# ---------------------------------------------------------------------------
class TestHazardRegistry:
    def test_list_active(self):
        active = HazardRegistry.list_active()
        assert len(active) > 0

    def test_get_hazard_ch27(self):
        h = HazardRegistry.get_hazard("CH27")
        assert isinstance(h, HazardStatus)
        assert h.status == "BLOCKED"

    def test_get_nonexistent(self):
        assert HazardRegistry.get_hazard("CH99") is None

    def test_all_active_codes_present(self):
        for code in ["CH27", "CH28", "CH31", "CH38", "CH46", "CH52"]:
            h = HazardRegistry.get_hazard(code)
            assert h is not None, f"{code} missing"

    def test_hazard_status_dataclass(self):
        h = HazardRegistry.get_hazard("CH52")
        assert hasattr(h, "mitigation")
        assert h.category == "ACTIVE"
