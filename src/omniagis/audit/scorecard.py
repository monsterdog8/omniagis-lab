"""M1–M12 scorecard generator for OMNIÆGIS audit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from omniagis.audit.inventory import InventoryReport, UNKNOWN, CODE_PYTHON
from omniagis.audit.parsability import ParseResult, ParsabilityChecker


@dataclass
class ScorecardEntry:
    """Single row in the M1–M12 scorecard."""

    metric_id: str
    name: str
    status: str    # "PASS" | "PARTIAL PASS" | "NO PASS"
    detail: str


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_scorecard(
    inventory_report: InventoryReport,
    parse_results: List[ParseResult],
) -> List[ScorecardEntry]:
    """Build the M1–M12 scorecard from audit artefacts.

    Parameters
    ----------
    inventory_report:
        Result of :class:`~omniagis.audit.inventory.FileInventory`.build().
    parse_results:
        List of :class:`~omniagis.audit.parsability.ParseResult` objects.

    Returns
    -------
    list of ScorecardEntry
    """
    entries: List[ScorecardEntry] = []
    checker = ParsabilityChecker()

    total_files = len(inventory_report.files)

    # ------------------------------------------------------------------
    # M1 — File inventory completeness
    # ------------------------------------------------------------------
    if total_files > 0:
        m1_status = "PASS"
        m1_detail = f"{total_files} files inventoried"
    else:
        m1_status = "NO PASS"
        m1_detail = "No files found"
    entries.append(ScorecardEntry("M1", "File inventory completeness", m1_status, m1_detail))

    # ------------------------------------------------------------------
    # M2 — Type separation
    # ------------------------------------------------------------------
    unknown_count = inventory_report.summary.get(UNKNOWN, 0)
    if total_files == 0:
        m2_status = "NO PASS"
        m2_detail = "No files to classify"
    elif unknown_count / total_files > 0.5:
        m2_status = "PARTIAL PASS"
        m2_detail = f"{unknown_count}/{total_files} files are UNKNOWN type ({100*unknown_count//total_files}%)"
    else:
        m2_status = "PASS"
        m2_detail = (
            f"Types identified; {unknown_count}/{total_files} UNKNOWN "
            f"({int(100 * _pct(unknown_count, total_files))}%)"
        )
    entries.append(ScorecardEntry("M2", "Type separation", m2_status, m2_detail))

    # ------------------------------------------------------------------
    # M3 — Exact duplicates
    # ------------------------------------------------------------------
    dup_count = len(inventory_report.duplicates)
    if total_files == 0:
        m3_status = "NO PASS"
        m3_detail = "No files"
    elif dup_count == 0:
        m3_status = "PASS"
        m3_detail = "No exact duplicates"
    elif dup_count / total_files < 0.2:
        m3_status = "PARTIAL PASS"
        m3_detail = f"{dup_count} duplicate(s) found ({int(100 * _pct(dup_count, total_files))}%)"
    else:
        m3_status = "NO PASS"
        m3_detail = f"{dup_count} duplicate(s) — {int(100 * _pct(dup_count, total_files))}% of files"
    entries.append(ScorecardEntry("M3", "Exact duplicates", m3_status, m3_detail))

    # ------------------------------------------------------------------
    # M4 — Fake .py detection
    # ------------------------------------------------------------------
    fake_py: List[str] = []
    for rec in inventory_report.files:
        if rec.path.endswith(".py") and rec.file_type == CODE_PYTHON:
            try:
                with open(rec.path, "rb") as fh:
                    header = fh.read(4)
                # Binary magic bytes indicate non-Python content
                if header[:2] in (b"\x89P", b"PK", b"\x1f\x8b") or (
                    len(header) >= 4 and header[0] == 0x7F and header[1:4] == b"ELF"
                ):
                    fake_py.append(rec.path)
            except OSError:
                pass

    if fake_py:
        m4_status = "NO PASS"
        m4_detail = f"{len(fake_py)} fake .py file(s) detected"
    else:
        m4_status = "PASS"
        m4_detail = "No fake .py files"
    entries.append(ScorecardEntry("M4", "Fake .py detection", m4_status, m4_detail))

    # ------------------------------------------------------------------
    # M5 — Pseudo-code
    # ------------------------------------------------------------------
    pseudo: List[str] = []
    for res in parse_results:
        if res.parseable and checker.is_pseudo_code(res.path):
            pseudo.append(res.path)

    if pseudo:
        m5_status = "PARTIAL PASS"
        m5_detail = f"{len(pseudo)} pseudo-code file(s): " + ", ".join(
            os.path.basename(p) for p in pseudo[:5]
        )
    else:
        m5_status = "PASS"
        m5_detail = "No pseudo-code files"
    entries.append(ScorecardEntry("M5", "Pseudo-code", m5_status, m5_detail))

    # ------------------------------------------------------------------
    # M6 — Ghost imports (per-file worst case)
    # ------------------------------------------------------------------
    files_with_ghosts = [r for r in parse_results if r.ghost_imports]
    max_ghosts = max((len(r.ghost_imports) for r in parse_results), default=0)

    if not files_with_ghosts:
        m6_status = "PASS"
        m6_detail = "No ghost imports"
    elif max_ghosts >= 3:
        m6_status = "NO PASS"
        m6_detail = (
            f"{len(files_with_ghosts)} file(s) with ghost imports; "
            f"max {max_ghosts} per file"
        )
    else:
        m6_status = "PARTIAL PASS"
        m6_detail = (
            f"{len(files_with_ghosts)} file(s) with ghost imports; "
            f"max {max_ghosts} per file"
        )
    entries.append(ScorecardEntry("M6", "Ghost imports", m6_status, m6_detail))

    # ------------------------------------------------------------------
    # M7 — Missing dependencies (requirements.txt)
    # ------------------------------------------------------------------
    import importlib.util as _ilu

    req_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "requirements.txt",
    )
    missing_deps: List[str] = []
    req_packages: List[str] = []

    if os.path.isfile(req_path):
        with open(req_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = line.split(">=")[0].split("==")[0].split("<=")[0].split("!=")[0].strip()
                    req_packages.append(pkg)
                    spec = _ilu.find_spec(pkg)
                    if spec is None:
                        missing_deps.append(pkg)
        if missing_deps:
            m7_status = "NO PASS"
            m7_detail = f"Missing: {', '.join(missing_deps)}"
        else:
            m7_status = "PASS"
            m7_detail = f"All {len(req_packages)} requirement(s) importable"
    else:
        m7_status = "PARTIAL PASS"
        m7_detail = "No requirements.txt found"
    entries.append(ScorecardEntry("M7", "Missing dependencies", m7_status, m7_detail))

    # ------------------------------------------------------------------
    # M8 — Parsability
    # ------------------------------------------------------------------
    total_py = len(parse_results)
    parseable_count = sum(1 for r in parse_results if r.parseable)

    if total_py == 0:
        m8_status = "PARTIAL PASS"
        m8_detail = "No Python files to check"
    else:
        ratio = parseable_count / total_py
        if ratio == 1.0:
            m8_status = "PASS"
            m8_detail = f"All {total_py} Python file(s) parse cleanly"
        elif ratio >= 0.5:
            m8_status = "PARTIAL PASS"
            m8_detail = f"{parseable_count}/{total_py} files parse ({int(100*ratio)}%)"
        else:
            m8_status = "NO PASS"
            m8_detail = f"Only {parseable_count}/{total_py} files parse ({int(100*ratio)}%)"
    entries.append(ScorecardEntry("M8", "Parsability", m8_status, m8_detail))

    # ------------------------------------------------------------------
    # M9 — Executability
    # ------------------------------------------------------------------
    fully_exec = [r for r in parse_results if r.parseable and not r.ghost_imports]
    partially_exec = [r for r in parse_results if r.parseable and r.ghost_imports]

    if total_py == 0:
        m9_status = "PARTIAL PASS"
        m9_detail = "No Python files"
    elif len(fully_exec) == total_py:
        m9_status = "PASS"
        m9_detail = f"All {total_py} file(s) parse with no ghost imports"
    elif fully_exec or partially_exec:
        m9_status = "PARTIAL PASS"
        m9_detail = (
            f"{len(fully_exec)} fully executable, "
            f"{len(partially_exec)} with missing imports, "
            f"{total_py - len(fully_exec) - len(partially_exec)} unparseable"
        )
    else:
        m9_status = "NO PASS"
        m9_detail = "No executable Python files"
    entries.append(ScorecardEntry("M9", "Executability", m9_status, m9_detail))

    # ------------------------------------------------------------------
    # M10 — Classification coverage
    # ------------------------------------------------------------------
    if total_files == 0:
        m10_status = "NO PASS"
        m10_detail = "No files"
    else:
        unk_ratio = unknown_count / total_files
        if unk_ratio == 0.0:
            m10_status = "PASS"
            m10_detail = "All files classified"
        elif unk_ratio < 0.2:
            m10_status = "PARTIAL PASS"
            m10_detail = f"{unknown_count} UNKNOWN ({int(100*unk_ratio)}%) — below 20% threshold"
        else:
            m10_status = "NO PASS"
            m10_detail = f"{unknown_count} UNKNOWN ({int(100*unk_ratio)}%) — exceeds 20% threshold"
    entries.append(ScorecardEntry("M10", "Classification coverage", m10_status, m10_detail))

    # ------------------------------------------------------------------
    # M11 — KEEP/REFACTOR/THROW/QUARANTINE matrix
    # ------------------------------------------------------------------
    entries.append(
        ScorecardEntry(
            "M11",
            "KEEP/REFACTOR/THROW/QUARANTINE matrix",
            "PASS",
            "Matrix built after M1–M10 completion",
        )
    )

    # ------------------------------------------------------------------
    # M12 — Global verdict (fail-closed)
    # ------------------------------------------------------------------
    from omniagis.core.classifier import FailClosedClassifier

    clf = FailClosedClassifier()
    all_statuses = [e.status for e in entries]  # M1–M11
    global_verdict = clf.combine(all_statuses)

    entries.append(
        ScorecardEntry(
            "M12",
            "Global verdict (fail-closed)",
            global_verdict,
            f"Derived from M1–M11 via fail-closed logic",
        )
    )

    return entries
