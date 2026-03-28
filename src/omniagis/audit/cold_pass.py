"""Mode MAVERICK cold-pass auditor — orchestrates outputs A–G."""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import List

from omniagis.audit.inventory import FileInventory, InventoryReport
from omniagis.audit.parsability import ParsabilityChecker, ParseResult
from omniagis.audit.scorecard import ScorecardEntry, build_scorecard
from omniagis.core.classifier import FailClosedClassifier


@dataclass
class ColdPassReport:
    """Full Mode MAVERICK audit report."""

    path: str
    structured_inventory: str    # A
    scorecard: str               # B
    file_audit: str              # C
    tensions: str                # D
    cleanup_plan: str            # E
    validation_plan: str         # F
    minimal_core: str            # G
    global_verdict: str

    # Structured data (not just strings) for programmatic access
    inventory_report: InventoryReport = field(repr=False, default=None)      # type: ignore[assignment]
    parse_results: List[ParseResult] = field(default_factory=list, repr=False)
    scorecard_entries: List[ScorecardEntry] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _table(headers: List[str], rows: List[List[str]], col_sep: str = "  ") -> str:
    """Render a plain-text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: List[str]) -> str:
        return col_sep.join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    sep = "-" * (sum(widths) + len(col_sep) * (len(headers) - 1))
    lines = [fmt_row(headers), sep]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append(fmt_row(padded))
    return "\n".join(lines)


def _rel(path: str, root: str) -> str:
    """Return *path* relative to *root* (fallback to basename)."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------

def _build_A(inventory: InventoryReport, root: str) -> str:
    headers = ["Path", "Type", "Size (B)", "SHA256 (short)", "Duplicate of"]
    rows: List[List[str]] = []
    for rec in inventory.files:
        rows.append([
            _rel(rec.path, root),
            rec.file_type,
            str(rec.size_bytes),
            rec.sha256[:12],
            _rel(rec.is_duplicate_of, root) if rec.is_duplicate_of else "",
        ])
    total = len(inventory.files)
    summary_lines = ["  " + f"{k}: {v}" for k, v in sorted(inventory.summary.items())]
    return (
        f"Total files: {total}\n"
        + "\n".join(summary_lines)
        + "\n\n"
        + _table(headers, rows)
    )


def _build_B(entries: List[ScorecardEntry]) -> str:
    headers = ["ID", "Metric", "Status", "Detail"]
    rows = [[e.metric_id, e.name, e.status, e.detail] for e in entries]
    return _table(headers, rows)


def _build_C(parse_results: List[ParseResult], root: str) -> str:
    headers = ["File", "Parseable", "Ghost imports", "Status"]
    rows: List[List[str]] = []
    for res in parse_results:
        ghosts = ", ".join(res.ghost_imports) if res.ghost_imports else "—"
        if not res.parseable:
            status = "NO PASS"
            extra = res.syntax_error or ""
        elif res.ghost_imports:
            status = "PARTIAL PASS"
            extra = ""
        else:
            status = "PASS"
            extra = ""
        rows.append([
            _rel(res.path, root),
            "yes" if res.parseable else f"NO ({extra})",
            ghosts,
            status,
        ])
    return _table(headers, rows) if rows else "(no Python files found)"


def _build_D(
    inventory: InventoryReport,
    parse_results: List[ParseResult],
    entries: List[ScorecardEntry],
) -> str:
    tensions: List[str] = []

    # Ghost imports in parseable files
    for res in parse_results:
        if res.parseable and res.ghost_imports:
            tensions.append(
                f"[GHOST-IMPORT] {os.path.basename(res.path)} references "
                f"unresolvable modules: {', '.join(res.ghost_imports)}"
            )

    # Duplicate files
    for a, b in inventory.duplicates:
        tensions.append(
            f"[DUPLICATE] {os.path.basename(a)} ≡ {os.path.basename(b)} (identical SHA-256)"
        )

    # Syntax errors
    for res in parse_results:
        if not res.parseable:
            tensions.append(
                f"[SYNTAX-ERROR] {os.path.basename(res.path)}: {res.syntax_error}"
            )

    # Scorecard failures
    failing = [e for e in entries if e.status == "NO PASS"]
    for e in failing:
        tensions.append(f"[SCORECARD-{e.metric_id}] {e.name}: {e.detail}")

    if not tensions:
        return "No unresolved tensions detected."
    return "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tensions))


def _build_E(
    inventory: InventoryReport,
    parse_results: List[ParseResult],
    root: str,
) -> str:
    keep: List[str] = []
    refactor: List[str] = []
    throw: List[str] = []
    quarantine: List[str] = []

    parseable_paths = {r.path for r in parse_results if r.parseable and not r.ghost_imports}
    ghost_paths = {r.path for r in parse_results if r.parseable and r.ghost_imports}
    broken_paths = {r.path for r in parse_results if not r.parseable}
    dup_paths = {b for _, b in inventory.duplicates}

    for rec in inventory.files:
        rel = _rel(rec.path, root)
        if rec.path in dup_paths:
            throw.append(rel)
        elif rec.path in broken_paths:
            quarantine.append(rel)
        elif rec.path in ghost_paths:
            refactor.append(rel)
        elif rec.path in parseable_paths:
            keep.append(rel)
        else:
            # non-Python files: keep docs/specs, quarantine unknowns
            from omniagis.audit.inventory import UNKNOWN, OUTPUT
            if rec.file_type in (UNKNOWN, OUTPUT):
                quarantine.append(rel)
            else:
                keep.append(rel)

    sections = []
    for label, items in [("KEEP", keep), ("REFACTOR", refactor), ("THROW", throw), ("QUARANTINE", quarantine)]:
        sections.append(f"[{label}] ({len(items)} files)")
        for item in sorted(items):
            sections.append(f"  • {item}")
    return "\n".join(sections)


def _build_F(entries: List[ScorecardEntry]) -> str:
    lines = [
        "Validation plan — steps to reach global PASS:",
        "",
        "1. Fix all syntax errors (M8) — run: python -m py_compile <file>",
        "2. Resolve ghost imports (M6, M9) — install missing packages or fix import paths",
        "3. Remove or merge duplicate files (M3)",
        "4. Add function/class definitions to pseudo-code files (M5)",
        "5. Classify UNKNOWN file types (M10)",
        "6. Run full test suite: pytest tests/ -v",
        "7. Re-run cold-pass audit: python -m omniagis.cli <path>",
        "8. Iterate until M12 = PASS",
    ]
    failing = [e for e in entries if e.status == "NO PASS"]
    if failing:
        lines.append("")
        lines.append("Priority failures to address:")
        for e in failing:
            lines.append(f"  [{e.metric_id}] {e.name} — {e.detail}")
    return "\n".join(lines)


def _build_G(
    inventory: InventoryReport,
    parse_results: List[ParseResult],
    root: str,
) -> str:
    core: List[str] = []
    parseable_no_ghost = {r.path for r in parse_results if r.parseable and not r.ghost_imports}

    for rec in inventory.files:
        from omniagis.audit.inventory import CODE_PYTHON, SPEC, DOC
        rel = _rel(rec.path, root)
        # Core = parseable Python + specs/configs + docs
        if rec.path in parseable_no_ghost:
            core.append(rel)
        elif rec.file_type in (SPEC, DOC) and rec.is_duplicate_of is None:
            core.append(rel)

    if not core:
        return "No minimal core identified (no parseable Python files found)."
    return "Minimal core to keep:\n" + "\n".join(f"  • {p}" for p in sorted(core))


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class ColdPass:
    """Mode MAVERICK cold-pass auditor."""

    def run(self, path: str) -> ColdPassReport:
        """Run the full Mode MAVERICK audit on *path*.

        Parameters
        ----------
        path:
            Absolute path to the root directory to audit.

        Returns
        -------
        ColdPassReport
        """
        path = os.path.abspath(path)

        inventory = FileInventory().build(path)
        parse_results = ParsabilityChecker().check_directory(path)
        entries = build_scorecard(inventory, parse_results)

        clf = FailClosedClassifier()
        global_verdict = clf.combine([e.status for e in entries])

        report = ColdPassReport(
            path=path,
            structured_inventory=_build_A(inventory, path),
            scorecard=_build_B(entries),
            file_audit=_build_C(parse_results, path),
            tensions=_build_D(inventory, parse_results, entries),
            cleanup_plan=_build_E(inventory, parse_results, path),
            validation_plan=_build_F(entries),
            minimal_core=_build_G(inventory, parse_results, path),
            global_verdict=global_verdict,
            inventory_report=inventory,
            parse_results=parse_results,
            scorecard_entries=entries,
        )
        return report

    def render(self, report: ColdPassReport) -> str:
        """Render *report* as a human-readable text string."""
        divider = "=" * 72
        thin = "-" * 72

        sections = [
            (
                "MODE MAVERICK — OMNIÆGIS COLD-PASS AUDIT",
                f"Target: {report.path}\nGlobal verdict: {report.global_verdict}",
            ),
            ("A. STRUCTURED INVENTORY", report.structured_inventory),
            ("B. SCORECARD M1–M12", report.scorecard),
            ("C. FILE-BY-FILE AUDIT", report.file_audit),
            ("D. UNRESOLVED TENSIONS", report.tensions),
            ("E. CLEANUP PLAN", report.cleanup_plan),
            ("F. VALIDATION PLAN", report.validation_plan),
            ("G. MINIMAL CORE TO KEEP", report.minimal_core),
        ]

        parts: List[str] = []
        for title, body in sections:
            parts.append(divider)
            parts.append(title)
            parts.append(thin)
            parts.append(body)
        parts.append(divider)
        parts.append(f"GLOBAL VERDICT: {report.global_verdict}")
        parts.append(divider)
        return "\n".join(parts)
