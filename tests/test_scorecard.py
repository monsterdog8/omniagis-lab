"""Boundary tests for M1–M12 scorecard generator."""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import pytest

from omniagis.audit.inventory import (
    FileRecord, InventoryReport,
    CODE_PYTHON, UNKNOWN, DOC, SPEC,
)
from omniagis.audit.parsability import ParseResult
from omniagis.audit.scorecard import ScorecardEntry, build_scorecard


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def make_record(path: str, file_type: str = CODE_PYTHON, size: int = 100,
                sha256: str = "abc", dup_of: Optional[str] = None) -> FileRecord:
    return FileRecord(path=path, file_type=file_type, size_bytes=size,
                      sha256=sha256, is_duplicate_of=dup_of)


def make_inventory(files: List[FileRecord] = None,
                   duplicates: List[Tuple[str, str]] = None,
                   summary: Dict[str, int] = None) -> InventoryReport:
    files = files or []
    duplicates = duplicates or []
    if summary is None:
        summary = {}
        for f in files:
            summary[f.file_type] = summary.get(f.file_type, 0) + 1
    return InventoryReport(files=files, duplicates=duplicates, summary=summary)


def make_parse(path: str, parseable: bool = True,
               ghost_imports: List[str] = None,
               syntax_error: Optional[str] = None) -> ParseResult:
    return ParseResult(
        path=path, parseable=parseable,
        syntax_error=syntax_error,
        imports=[], ghost_imports=ghost_imports or [],
    )


def get_metric(entries: List[ScorecardEntry], mid: str) -> ScorecardEntry:
    for e in entries:
        if e.metric_id == mid:
            return e
    raise KeyError(mid)


# ---------------------------------------------------------------------------
# M1 — File inventory completeness
# ---------------------------------------------------------------------------

class TestM1:
    def test_pass_when_files_present(self, tmp_path):
        inv = make_inventory([make_record(str(tmp_path / "a.py"))])
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M1").status == "PASS"

    def test_no_pass_when_empty(self):
        inv = make_inventory([])
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M1").status == "NO PASS"

    def test_detail_includes_count(self, tmp_path):
        files = [make_record(str(tmp_path / f"f{i}.py")) for i in range(3)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert "3" in get_metric(entries, "M1").detail


# ---------------------------------------------------------------------------
# M2 — Type separation
# ---------------------------------------------------------------------------

class TestM2:
    def test_pass_when_no_unknown(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M2").status == "PASS"

    def test_partial_when_over_50_pct_unknown(self, tmp_path):
        # 3 unknown out of 4 → 75% > 50%
        files = [
            make_record(str(tmp_path / "a.py"), CODE_PYTHON),
            make_record(str(tmp_path / "b.xyz"), UNKNOWN, sha256="h1"),
            make_record(str(tmp_path / "c.xyz"), UNKNOWN, sha256="h2"),
            make_record(str(tmp_path / "d.xyz"), UNKNOWN, sha256="h3"),
        ]
        inv = make_inventory(files, summary={CODE_PYTHON: 1, UNKNOWN: 3})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M2").status == "PARTIAL PASS"

    def test_pass_when_exactly_50_pct_unknown(self, tmp_path):
        # 1/2 = 50% — NOT > 0.5, so PASS
        files = [
            make_record(str(tmp_path / "a.py"), CODE_PYTHON),
            make_record(str(tmp_path / "b.xyz"), UNKNOWN, sha256="h1"),
        ]
        inv = make_inventory(files, summary={CODE_PYTHON: 1, UNKNOWN: 1})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M2").status == "PASS"

    def test_no_pass_when_no_files(self):
        inv = make_inventory([], summary={})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M2").status == "NO PASS"


# ---------------------------------------------------------------------------
# M3 — Exact duplicates
# ---------------------------------------------------------------------------

class TestM3:
    def test_pass_when_no_duplicates(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), sha256="aaa"),
                 make_record(str(tmp_path / "b.py"), sha256="bbb")]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M3").status == "PASS"

    def test_partial_when_dup_ratio_under_20_pct(self, tmp_path):
        # 1 dup out of 10 files = 10% < 20%
        files = [make_record(str(tmp_path / f"f{i}.py"), sha256=f"h{i}") for i in range(10)]
        inv = make_inventory(files, duplicates=[(str(tmp_path / "f0.py"), str(tmp_path / "f1.py"))])
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M3").status == "PARTIAL PASS"

    def test_no_pass_when_dup_ratio_over_20_pct(self, tmp_path):
        # 3 dups out of 5 files = 60% >= 20%
        files = [make_record(str(tmp_path / f"f{i}.py"), sha256=f"h{i}") for i in range(5)]
        dups = [(str(tmp_path / f"f{i}.py"), str(tmp_path / f"f{i+1}.py")) for i in range(3)]
        inv = make_inventory(files, duplicates=dups)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M3").status == "NO PASS"

    def test_no_pass_when_no_files(self):
        inv = make_inventory([], duplicates=[])
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M3").status == "NO PASS"


# ---------------------------------------------------------------------------
# M4 — Fake .py detection
# ---------------------------------------------------------------------------

class TestM4:
    def test_pass_when_real_python(self, tmp_path):
        p = tmp_path / "real.py"
        p.write_text("def foo(): pass\n")
        files = [make_record(str(p), CODE_PYTHON)]
        inv = make_inventory(files)
        parse = [make_parse(str(p))]
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M4").status == "PASS"

    def test_no_pass_when_png_magic_in_py(self, tmp_path):
        p = tmp_path / "fake.py"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data")
        files = [make_record(str(p), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M4").status == "NO PASS"

    def test_no_pass_when_zip_magic_in_py(self, tmp_path):
        p = tmp_path / "fake.py"
        p.write_bytes(b"PK\x03\x04" + b"data")
        files = [make_record(str(p), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M4").status == "NO PASS"

    def test_no_pass_when_elf_magic_in_py(self, tmp_path):
        p = tmp_path / "fake.py"
        p.write_bytes(b"\x7fELF\x02" + b"data")
        files = [make_record(str(p), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M4").status == "NO PASS"

    def test_no_pass_when_gzip_magic_in_py(self, tmp_path):
        p = tmp_path / "fake.py"
        p.write_bytes(b"\x1f\x8b" + b"\x00" * 10)
        files = [make_record(str(p), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M4").status == "NO PASS"


# ---------------------------------------------------------------------------
# M5 — Pseudo-code
# ---------------------------------------------------------------------------

class TestM5:
    def test_pass_when_no_pseudo(self, tmp_path):
        p = tmp_path / "real.py"
        p.write_text("def foo(): pass\n")
        inv = make_inventory([make_record(str(p), CODE_PYTHON)])
        parse = [make_parse(str(p))]
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M5").status == "PASS"

    def test_partial_when_pseudo_code_present(self, tmp_path):
        p = tmp_path / "pseudo.py"
        p.write_text("# just a comment\n")  # parseable, no defs
        inv = make_inventory([make_record(str(p), CODE_PYTHON)])
        parse = [make_parse(str(p))]
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M5").status == "PARTIAL PASS"

    def test_unparseable_file_not_counted_as_pseudo(self, tmp_path):
        p = tmp_path / "broken.py"
        p.write_text("def foo(:\n    pass\n")
        inv = make_inventory([make_record(str(p), CODE_PYTHON)])
        parse = [make_parse(str(p), parseable=False, syntax_error="SyntaxError")]
        entries = build_scorecard(inv, parse)
        # broken file is not parseable → not checked for pseudo-code
        assert get_metric(entries, "M5").status == "PASS"


# ---------------------------------------------------------------------------
# M6 — Ghost imports
# ---------------------------------------------------------------------------

class TestM6:
    def test_pass_when_no_ghosts(self, tmp_path):
        p = str(tmp_path / "a.py")
        parse = [make_parse(p, ghost_imports=[])]
        inv = make_inventory([make_record(p)])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M6").status == "PASS"

    def test_partial_when_max_ghosts_under_3(self, tmp_path):
        p = str(tmp_path / "a.py")
        parse = [make_parse(p, ghost_imports=["fake1", "fake2"])]
        inv = make_inventory([make_record(p)])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M6").status == "PARTIAL PASS"

    def test_no_pass_when_max_ghosts_3_or_more(self, tmp_path):
        p = str(tmp_path / "a.py")
        parse = [make_parse(p, ghost_imports=["fake1", "fake2", "fake3"])]
        inv = make_inventory([make_record(p)])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M6").status == "NO PASS"

    def test_boundary_exactly_3_is_no_pass(self, tmp_path):
        p = str(tmp_path / "a.py")
        parse = [make_parse(p, ghost_imports=["f1", "f2", "f3"])]
        inv = make_inventory([make_record(p)])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M6").status == "NO PASS"


# ---------------------------------------------------------------------------
# M8 — Parsability
# ---------------------------------------------------------------------------

class TestM8:
    def test_pass_all_parseable(self, tmp_path):
        paths = [str(tmp_path / f"f{i}.py") for i in range(3)]
        parse = [make_parse(p) for p in paths]
        inv = make_inventory([make_record(p) for p in paths])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M8").status == "PASS"

    def test_partial_when_between_50_and_100_pct(self, tmp_path):
        good = str(tmp_path / "good.py")
        bad = str(tmp_path / "bad.py")
        parse = [make_parse(good), make_parse(bad, parseable=False, syntax_error="err")]
        inv = make_inventory([make_record(good), make_record(bad)])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M8").status == "PARTIAL PASS"

    def test_no_pass_when_under_50_pct(self, tmp_path):
        paths = [str(tmp_path / f"f{i}.py") for i in range(4)]
        parse = [make_parse(paths[0])] + [
            make_parse(p, parseable=False, syntax_error="err") for p in paths[1:]
        ]
        inv = make_inventory([make_record(p) for p in paths])
        entries = build_scorecard(inv, parse)
        assert get_metric(entries, "M8").status == "NO PASS"

    def test_partial_when_no_python_files(self):
        inv = make_inventory([make_record("/tmp/readme.md", DOC)])
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M8").status == "PARTIAL PASS"


# ---------------------------------------------------------------------------
# M10 — Classification coverage
# ---------------------------------------------------------------------------

class TestM10:
    def test_pass_when_all_classified(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), CODE_PYTHON)]
        inv = make_inventory(files, summary={CODE_PYTHON: 1})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M10").status == "PASS"

    def test_partial_when_unknown_under_20_pct(self, tmp_path):
        # 1/10 = 10% → PARTIAL PASS (below threshold)
        files = [make_record(str(tmp_path / f"f{i}.py"), CODE_PYTHON, sha256=f"h{i}") for i in range(9)]
        files.append(make_record(str(tmp_path / "unk.xyz"), UNKNOWN, sha256="hunk"))
        inv = make_inventory(files, summary={CODE_PYTHON: 9, UNKNOWN: 1})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M10").status == "PARTIAL PASS"

    def test_no_pass_when_unknown_over_20_pct(self, tmp_path):
        # 3/10 = 30% → NO PASS
        files = [make_record(str(tmp_path / f"f{i}.py"), CODE_PYTHON, sha256=f"h{i}") for i in range(7)]
        files += [make_record(str(tmp_path / f"u{i}.xyz"), UNKNOWN, sha256=f"hu{i}") for i in range(3)]
        inv = make_inventory(files, summary={CODE_PYTHON: 7, UNKNOWN: 3})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M10").status == "NO PASS"

    def test_no_pass_when_no_files(self):
        inv = make_inventory([], summary={})
        entries = build_scorecard(inv, [])
        assert get_metric(entries, "M10").status == "NO PASS"


# ---------------------------------------------------------------------------
# M12 — Global verdict (fail-closed)
# ---------------------------------------------------------------------------

class TestM12:
    def test_m12_is_last_entry(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert entries[-1].metric_id == "M12"

    def test_m12_no_pass_when_any_no_pass(self):
        # empty directory guarantees several NO PASS metrics
        inv = make_inventory([], summary={})
        entries = build_scorecard(inv, [])
        m12 = get_metric(entries, "M12")
        assert m12.status == "NO PASS"

    def test_scorecard_has_12_entries(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        assert len(entries) == 12

    def test_all_metric_ids_present(self, tmp_path):
        files = [make_record(str(tmp_path / "a.py"), CODE_PYTHON)]
        inv = make_inventory(files)
        entries = build_scorecard(inv, [])
        ids = {e.metric_id for e in entries}
        assert ids == {f"M{i}" for i in range(1, 13)}
