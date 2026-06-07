"""Minimal usage examples for MODULE_003_EVIDENCE.

Run from repository root:
    python modules/MODULE_003_EVIDENCE/examples/example.py
"""
from __future__ import annotations

import sys
import pathlib
import json
import zipfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from evidence import (
    sha256_text, sha256_json, canonical_json,
    build_raw_record, validate_raw_record, validate_raw_records,
    audit_zip, summarize_csv, load_jsonl,
)


# ---------------------------------------------------------------------------
# Example 1: Raw record construction, validation, and batch gate
# ---------------------------------------------------------------------------

def example_raw_records():
    print("=== Example 1: Raw record construction and batch gate ===")

    # Build 3 raw records
    records = [
        build_raw_record(
            run_id="run_demo",
            task_id=f"task_{i:03d}",
            output_id=f"out_{i:03d}",
            model_slot="model_A",
            raw_output=f"The model output for task {i}.",
            token_count=12 + i,
            latency_ms=200 + i * 10,
        )
        for i in range(3)
    ]

    # Validate each individually
    for rec in records:
        verdict = validate_raw_record(rec)
        print(f"  {verdict.output_id}: valid={verdict.valid}, errors={verdict.errors}")

    # Batch gate
    report = validate_raw_records(run_id="run_demo", expected=3, records=records)
    print(f"  pass_gate       : {report.pass_gate}")
    print(f"  scoring_status  : {report.scoring_status}")
    print(f"  verdict         : {report.verdict}")
    print()


# ---------------------------------------------------------------------------
# Example 2: SHA-256 hashing helpers
# ---------------------------------------------------------------------------

def example_hashing():
    print("=== Example 2: SHA-256 hashing helpers ===")

    h_text = sha256_text("hello world")
    print(f"  sha256_text('hello world') : {h_text}")

    obj = {"metric": "accuracy", "value": 0.92, "cycle": 7}
    canon = canonical_json(obj)
    h_json = sha256_json(obj)
    print(f"  canonical_json(obj)        : {canon}")
    print(f"  sha256_json(obj)           : {h_json}")
    print()


# ---------------------------------------------------------------------------
# Example 3: ZIP audit and CSV summary on temporary files
# ---------------------------------------------------------------------------

def example_audit_zip_and_csv():
    print("=== Example 3: ZIP audit and CSV summary ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # Create a small ZIP
        zip_path = tmp / "run_artifact.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("outputs/out_000.txt", "model output text")
            z.writestr("run_summary.json", json.dumps({
                "ledger_replay": {"status": "PASS", "mismatch_count": 0}
            }))

        result = audit_zip(zip_path)
        print(f"  ZIP exists      : {result['exists']}")
        print(f"  member_count    : {result['member_count']}")
        print(f"  replay_pass     : {result['replay_pass']}")
        print(f"  verdict         : {result['verdict']}")
        print(f"  proof_status    : {result['proof_status']}")

        # Create a CSV and summarize
        csv_path = tmp / "metrics.csv"
        csv_path.write_text("metric,value\naccuracy,0.88\nprecision,0.91\nrecall,0.85\n")
        summary = summarize_csv(csv_path)
        print(f"  CSV rows        : {summary['rows']}")
        print(f"  numeric_cells   : {summary['numeric_cells']}")
        print(f"  min/max/mean    : {summary['min']}/{summary['max']}/{summary['mean']:.4f}")

    print()


if __name__ == "__main__":
    example_raw_records()
    example_hashing()
    example_audit_zip_and_csv()
