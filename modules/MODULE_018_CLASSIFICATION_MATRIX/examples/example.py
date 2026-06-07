"""Minimal usage examples for MODULE_018_CLASSIFICATION_MATRIX.

Demonstrates how to load the schema template, fill in a classification entry,
and check C08 gate threshold logic.

Run from repository root:
    python modules/MODULE_018_CLASSIFICATION_MATRIX/examples/example.py
"""
from __future__ import annotations

import json
import pathlib
import copy

_EXPORTS_DIR = pathlib.Path(__file__).parent.parent / "exports"
_SCHEMA_FILE = _EXPORTS_DIR / "classification_matrix_schema_v1.json"


# ---------------------------------------------------------------------------
# Example 1: Load the schema and inspect structure
# ---------------------------------------------------------------------------

def example_load_schema():
    print("=== Example 1: Load classification_matrix_schema_v1.json ===")
    with _SCHEMA_FILE.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    print(f"  Schema       : {schema['_schema']}")
    print(f"  Version      : {schema['_version']}")
    print(f"  Status       : {schema['_status']}")
    print(f"  Claim ceiling: {schema['_claim_ceiling']}")
    print()
    print(f"  Auditors ({len(schema['auditors'])}):")
    for a in schema["auditors"]:
        print(f"    - {a['auditor_id']} ({a['auditor_type']}, blind={a['blind']})")
    print()
    print(f"  Label values: {schema['label_schema']['values']}")
    print(f"  Confidence range: {schema['label_schema']['confidence_range']}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Fill in a classification entry
# ---------------------------------------------------------------------------

def example_fill_classification_entry():
    print("=== Example 2: Fill in a classification entry ===")
    with _SCHEMA_FILE.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    # Simulate filling in one entry for HUMAN_01 on claim C03
    entry = {
        "auditor_id": "HUMAN_01",
        "claim_id": "C03",
        "claim_hash": "abc123def456" + "0" * 52,  # placeholder hash
        "label": "SUPPORTED",
        "confidence": 8,
        "rationale_brief": "The evidence in the repo supports this claim within its stated scope.",
        "timestamp": "2026-06-07T12:00:00Z",
    }

    # Validate required fields
    required_fields = set(schema["label_schema"]["required_fields"])
    entry_keys = set(entry.keys())
    missing = required_fields - entry_keys
    valid_label = entry["label"] in schema["label_schema"]["values"]
    valid_confidence = (
        schema["label_schema"]["confidence_range"][0]
        <= entry["confidence"]
        <= schema["label_schema"]["confidence_range"][1]
    )

    print(f"  Entry filled:")
    for k, v in entry.items():
        print(f"    {k}: {v}")
    print()
    print(f"  Required fields satisfied : {not missing} (missing: {missing or 'none'})")
    print(f"  Valid label               : {valid_label}")
    print(f"  Valid confidence [0,10]   : {valid_confidence}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Check C08 gate threshold for a simulated Krippendorff alpha
# ---------------------------------------------------------------------------

def example_check_c08_gate():
    print("=== Example 3: C08 gate threshold check ===")
    with _SCHEMA_FILE.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    gate = schema["c08_gate"]
    min_threshold = gate["alpha_threshold_minimum"]
    sub_threshold = gate["alpha_threshold_substantial"]

    simulated_alphas = [0.45, 0.667, 0.72, 0.85]
    print(f"  Minimum threshold   : {min_threshold}")
    print(f"  Substantial threshold: {sub_threshold}")
    print()
    print(f"  {'Alpha':>8}  {'>= min':>8}  {'>= substantial':>16}  {'C08 verdict':>22}")
    for alpha in simulated_alphas:
        meets_min = alpha >= min_threshold
        meets_sub = alpha >= sub_threshold
        if meets_sub:
            verdict = "SUBSTANTIAL_AGREEMENT"
        elif meets_min:
            verdict = "MINIMUM_AGREEMENT"
        else:
            verdict = "GATE_FAIL__BLOCKED"
        blocked = ", ".join(gate["blocked_claims_on_fail"]) if not meets_min else "none"
        print(f"  {alpha:>8.3f}  {str(meets_min):>8}  {str(meets_sub):>16}  {verdict:>22}")
        if not meets_min:
            print(f"           Blocked claims: {blocked}")
    print()


if __name__ == "__main__":
    example_load_schema()
    example_fill_classification_entry()
    example_check_c08_gate()
