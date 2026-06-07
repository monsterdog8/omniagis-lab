"""Minimal usage examples for MODULE_015_CLI.

Demonstrates calling the CLI programmatically via main() and via build_parser().

Run from repository root:
    python modules/MODULE_015_CLI/examples/example.py
"""
from __future__ import annotations

import json
import sys
import pathlib
import tempfile
import io

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from cli import build_parser, main


# ---------------------------------------------------------------------------
# Example 1: Programmatic classify-claim via main()
# ---------------------------------------------------------------------------

def example_classify_claim():
    print("=== Example 1: classify-claim via main() ===")

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    exit_code = main(["classify-claim", "This is a lab-only prototype hypothesis"])
    sys.stdout = old_stdout

    output = buf.getvalue().strip()
    result = json.loads(output)
    print(f"  Input   : 'This is a lab-only prototype hypothesis'")
    print(f"  Status  : {result.get('status')}")
    print(f"  Exit    : {exit_code}")
    print()


# ---------------------------------------------------------------------------
# Example 2: evidence-score via main()
# ---------------------------------------------------------------------------

def example_evidence_score():
    print("=== Example 2: evidence-score via main() ===")

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    exit_code = main([
        "evidence-score",
        "--expected", "10",
        "--present", "10",
        "--valid", "10",
        "--safety", "1.0",
        "--scoring",
        "--replay",
        "--independence",
    ])
    sys.stdout = old_stdout

    result = json.loads(buf.getvalue().strip())
    print(f"  final_score          : {result.get('final_score')}")
    print(f"  public_claim_allowed : {result.get('public_claim_allowed')}")
    print(f"  Exit                 : {exit_code}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Inspect parser subcommands via build_parser()
# ---------------------------------------------------------------------------

def example_inspect_parser():
    print("=== Example 3: inspect parser subcommands via build_parser() ===")
    parser = build_parser()
    subparsers_action = None
    for action in parser._actions:
        if hasattr(action, '_name_parser_map'):
            subparsers_action = action
            break
    if subparsers_action:
        names = sorted(subparsers_action._name_parser_map.keys())
        print(f"  Registered subcommands ({len(names)}):")
        for name in names:
            print(f"    - {name}")
    print()


if __name__ == "__main__":
    example_classify_claim()
    example_evidence_score()
    example_inspect_parser()
