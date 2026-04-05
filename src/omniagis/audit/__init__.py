"""OMNIÆGIS audit components.

GO OMNIÆGIS Audit Module — Mode MAVERICK
==========================================

This module provides comprehensive code quality auditing capabilities for Python
projects using the Mode MAVERICK protocol (M1–M12 scorecard + outputs A–G).

Components:
-----------
FileInventory:
    Recursively scans directories, classifies file types (CODE_PYTHON, DOC, SPEC,
    OUTPUT, UNKNOWN), computes SHA-256 hashes, and detects exact duplicates.

ParsabilityChecker:
    Validates Python files for syntactic correctness and import resolvability.
    Identifies ghost imports (unresolvable modules), syntax errors, and pseudo-code
    files (no function/class definitions).

build_scorecard:
    Generates M1–M12 metrics scorecard combining inventory, parsability, and
    dependency analysis results. Returns list of ScorecardEntry objects with
    PASS/PARTIAL PASS/NO PASS verdicts.

ColdPass:
    Mode MAVERICK orchestrator. Coordinates all audit components to produce
    7-section output (A–G):
        A. Structured inventory (all files with type, size, SHA-256)
        B. Scorecard M1–M12 (metrics table)
        C. File-by-file audit (per-file parsability + import status)
        D. Unresolved tensions (ghost imports, duplicates, syntax errors)
        E. Cleanup plan (KEEP/REFACTOR/THROW/QUARANTINE matrix)
        F. Validation plan (steps to reach global PASS)
        G. Minimal core (essential files to keep)

M1–M12 Metrics:
---------------
    M1:  File inventory completeness
    M2:  Type separation (< 50% UNKNOWN)
    M3:  Exact duplicates (SHA-256)
    M4:  Fake .py detection (binary files with .py extension)
    M5:  Pseudo-code files (parse but no definitions)
    M6:  Ghost imports (unresolvable)
    M7:  Missing dependencies (requirements.txt vs. installed)
    M8:  Parsability (valid syntax)
    M9:  Executability (parse + no ghost imports)
    M10: Classification coverage (< 20% UNKNOWN)
    M11: KEEP/REFACTOR/THROW/QUARANTINE matrix
    M12: Global verdict (fail-closed from M1–M11)

Usage:
------
    >>> from omniagis.audit import ColdPass
    >>> auditor = ColdPass()
    >>> report = auditor.run("/path/to/project")
    >>> print(report.global_verdict)  # "PASS" | "PARTIAL PASS" | "NO PASS"
    >>> print(auditor.render(report))  # Full A–G report

    >>> from omniagis.audit import FileInventory, ParsabilityChecker, build_scorecard
    >>> inventory = FileInventory().build("/path")
    >>> parse_results = ParsabilityChecker().check_directory("/path")
    >>> scorecard = build_scorecard(inventory, parse_results)
    >>> for entry in scorecard:
    ...     print(f"{entry.metric_id}: {entry.status} — {entry.detail}")
"""

from .inventory import FileInventory
from .parsability import ParsabilityChecker
from .scorecard import build_scorecard
from .cold_pass import ColdPass

__all__ = ["FileInventory", "ParsabilityChecker", "build_scorecard", "ColdPass"]
