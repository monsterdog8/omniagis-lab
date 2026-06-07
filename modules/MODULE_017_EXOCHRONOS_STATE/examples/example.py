"""Minimal usage examples for MODULE_017_EXOCHRONOS_STATE.

Demonstrates how to load and read the EXOCHRONOS system state snapshot.

Run from repository root:
    python modules/MODULE_017_EXOCHRONOS_STATE/examples/example.py
"""
from __future__ import annotations

import json
import pathlib

_EXPORTS_DIR = pathlib.Path(__file__).parent.parent / "exports"
_STATE_FILE = _EXPORTS_DIR / "EXOCHRONOS_STATE_v1.json"


# ---------------------------------------------------------------------------
# Example 1: Load the state and print top-level metadata
# ---------------------------------------------------------------------------

def example_load_and_metadata():
    print("=== Example 1: Load EXOCHRONOS_STATE_v1.json and read metadata ===")
    with _STATE_FILE.open("r", encoding="utf-8") as f:
        state = json.load(f)

    print(f"  Schema         : {state['_schema']}")
    print(f"  Version        : {state['_version']}")
    print(f"  Date           : {state['_date']}")
    print(f"  Claim ceiling  : {state['_claim_ceiling']}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Read component readiness and gap assessment
# ---------------------------------------------------------------------------

def example_component_status():
    print("=== Example 2: Component readiness and gap assessment ===")
    with _STATE_FILE.open("r", encoding="utf-8") as f:
        state = json.load(f)

    print("  Component status:")
    for component, status in state["component_status"].items():
        marker = "OK" if status == "READY" else "!!"
        print(f"    [{marker}] {component}: {status}")

    print()
    print("  Gap assessment:")
    for gap, level in state["gap_assessment"].items():
        print(f"    {gap}: {level}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Read packet status and owner decision point
# ---------------------------------------------------------------------------

def example_packet_and_decision():
    print("=== Example 3: Packet status and owner decision point ===")
    with _STATE_FILE.open("r", encoding="utf-8") as f:
        state = json.load(f)

    pkt = state["packet"]
    print(f"  Packet ID      : {pkt['id']}")
    print(f"  Packet hash    : {pkt['packet_hash'][:16]}...{pkt['packet_hash'][-8:]}")
    print(f"  Frozen         : {pkt['frozen']}")
    print(f"  Canonical      : {pkt['canonical']}")
    print(f"  Status         : {pkt['canonical_status']}")
    print()

    odp = state["owner_decision_point"]
    print(f"  Option A : {odp['option_a']}")
    print(f"  Option B : {odp['option_b']}")
    print(f"  Rational default: {odp['rational_default']}")
    print()

    print(f"  Next gate      : {state['next_phase']['gate']}")
    print(f"  Blocked until  : {state['next_phase']['blocked_until']}")
    print()


if __name__ == "__main__":
    example_load_and_metadata()
    example_component_status()
    example_packet_and_decision()
