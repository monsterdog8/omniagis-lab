"""CLI entry point: python -m omniagis.cli <path>"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from omniagis.audit.cold_pass import ColdPass


def _verdict_exit_code(verdict: str) -> int:
    if verdict == "PASS":
        return 0
    if verdict == "PARTIAL PASS":
        return 1
    return 2  # NO PASS or unknown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="omniagis",
        description="OMNIÆGIS Mode MAVERICK cold-pass auditor",
    )
    parser.add_argument(
        "path",
        help="Path to the directory to audit",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args(argv)

    cp = ColdPass()
    report = cp.run(args.path)

    if args.output == "json":
        data = {
            "path": report.path,
            "global_verdict": report.global_verdict,
            "structured_inventory": report.structured_inventory,
            "scorecard": report.scorecard,
            "file_audit": report.file_audit,
            "tensions": report.tensions,
            "cleanup_plan": report.cleanup_plan,
            "validation_plan": report.validation_plan,
            "minimal_core": report.minimal_core,
        }
        print(json.dumps(data, indent=2))
    else:
        print(cp.render(report))

    sys.exit(_verdict_exit_code(report.global_verdict))


if __name__ == "__main__":
    main()
