"""Minimal usage examples for MODULE_005_LEDGER.

Run from repository root:
    python modules/MODULE_005_LEDGER/examples/example.py
"""
from __future__ import annotations

import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from ledger import AuditLedger, ContextCitationLock, _SyncGate


# ---------------------------------------------------------------------------
# Example 1: In-memory chained audit ledger
# ---------------------------------------------------------------------------

def example_chained_ledger():
    print("=== Example 1: In-memory chained audit ledger ===")

    ledger = AuditLedger(chained=True)

    # Log a sequence of events
    e1 = ledger.log("RUN_START",   {"run_id": "demo_001", "cycle": 7}, audit=True)
    e2 = ledger.log("RAW_CAPTURE", {"records": 10, "valid": 10},        audit=True)
    e3 = ledger.log("SCORE_DONE",  {"score": 0.88, "status": "LOCAL"},  audit=True)

    print(f"  entry 1 hash: {e1['entry_hash'][:32]}...")
    print(f"  entry 2 prev: {e2['prev_hash'][:32]}...")
    print(f"  entry 3 hash: {e3['entry_hash'][:32]}...")

    # Verify chain integrity
    result = ledger.verify_chain()
    print(f"  chain valid       : {result['valid']}")
    print(f"  entries_checked   : {result.get('entries_checked')}")
    print(f"  total entries     : {len(ledger.entries())}")
    print()


# ---------------------------------------------------------------------------
# Example 2: File-persisted ledger with reload
# ---------------------------------------------------------------------------

def example_file_ledger():
    print("=== Example 2: File-persisted ledger with reload ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        p = pathlib.Path(tmpdir) / "audit.jsonl"

        # Write
        ledger_w = AuditLedger(path=p, chained=True)
        ledger_w.log("GATE_A", {"step": "raw_collection"},   audit=True)
        ledger_w.log("GATE_B", {"step": "scoring_complete"}, audit=True)
        print(f"  written {len(ledger_w.entries())} entries to {p.name}")

        # Reload and verify
        ledger_r = AuditLedger(path=p, chained=True)
        result = ledger_r.verify_chain()
        print(f"  reloaded entries  : {len(ledger_r.entries())}")
        print(f"  chain valid       : {result['valid']}")
    print()


# ---------------------------------------------------------------------------
# Example 3: ContextCitationLock proof-of-citation
# ---------------------------------------------------------------------------

def example_citation_lock():
    print("=== Example 3: ContextCitationLock proof-of-citation ===")

    lock = ContextCitationLock()

    # Simulated corpus (could be streaming chunks from a large document)
    corpus_chunks = [
        "The run produced 10 raw outputs, all hash-valid.",
        "Final structure_score = 0.7312 on cycle C007.",
        "Ledger chain verified with 0 mismatches.",
    ]

    # Check citations
    checks = [
        ("structure_score = 0.7312",    True),
        ("structure_score = 0.9999",    False),
        ("10 raw outputs, all hash-valid", True),
        ("production-ready",            False),
    ]

    for value, expected in checks:
        found = lock.has_citation(corpus_chunks, value)
        status = "OK" if found == expected else "UNEXPECTED"
        print(f"  [{status}] has_citation({value!r}) = {found}")

    # require_citation raises on missing value
    try:
        lock.require_citation(corpus_chunks, "NOT IN CORPUS")
        print("  ERROR: should have raised")
    except ValueError as e:
        print(f"  require_citation raised as expected: {e}")

    # require_citation passes on present value
    lock.require_citation(corpus_chunks, "10 raw outputs, all hash-valid")
    print("  require_citation('10 raw outputs...') passed")
    print()


if __name__ == "__main__":
    example_chained_ledger()
    example_file_ledger()
    example_citation_lock()
