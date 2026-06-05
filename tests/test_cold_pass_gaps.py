"""Coverage gaps for cold_pass.py — _rel ValueError, ghost-import branches
in _build_C / _build_D / _build_E, and UNKNOWN/OUTPUT quarantine in _build_E."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omniagis.audit.cold_pass import (
    ColdPass,
    _build_C,
    _build_D,
    _build_E,
    _rel,
)
from omniagis.audit.inventory import (
    FileRecord,
    InventoryReport,
    OUTPUT,
    UNKNOWN,
    CODE_PYTHON,
)
from omniagis.audit.parsability import ParseResult
from omniagis.audit.scorecard import ScorecardEntry


# ---------------------------------------------------------------------------
# _rel — ValueError fallback (os.path.relpath raises on Windows cross-drive)
# ---------------------------------------------------------------------------

class TestRelFallback:
    def test_value_error_falls_back_to_basename(self, monkeypatch):
        def raise_ve(path, root):
            raise ValueError("cross-drive path")
        monkeypatch.setattr("omniagis.audit.cold_pass.os.path.relpath", raise_ve)
        assert _rel("/C/path/to/file.py", "/D/root") == "file.py"

    def test_normal_relpath_works(self):
        result = _rel("/root/sub/file.py", "/root")
        assert result == os.path.join("sub", "file.py")


# ---------------------------------------------------------------------------
# _build_C — ghost imports branch (PARTIAL PASS row in the audit table)
# ---------------------------------------------------------------------------

def _make_parse(path, parseable=True, ghost_imports=None, syntax_error=None):
    return ParseResult(
        path=path, parseable=parseable,
        syntax_error=syntax_error,
        imports=[], ghost_imports=ghost_imports or [],
    )


class TestBuildCGhostImports:
    def test_ghost_import_row_status_is_partial_pass(self, tmp_path):
        p = str(tmp_path / "ghosty.py")
        res = _make_parse(p, parseable=True, ghost_imports=["fake_pkg"])
        output = _build_C([res], str(tmp_path))
        assert "PARTIAL PASS" in output

    def test_ghost_import_row_shows_module_name(self, tmp_path):
        p = str(tmp_path / "ghosty.py")
        res = _make_parse(p, parseable=True, ghost_imports=["fake_pkg"])
        output = _build_C([res], str(tmp_path))
        assert "fake_pkg" in output

    def test_multiple_ghost_imports_joined(self, tmp_path):
        p = str(tmp_path / "multi.py")
        res = _make_parse(p, parseable=True, ghost_imports=["alpha", "beta"])
        output = _build_C([res], str(tmp_path))
        assert "alpha" in output
        assert "beta" in output

    def test_clean_file_shows_pass(self, tmp_path):
        p = str(tmp_path / "clean.py")
        res = _make_parse(p, parseable=True, ghost_imports=[])
        output = _build_C([res], str(tmp_path))
        assert "PASS" in output

    def test_unparseable_shows_no_pass(self, tmp_path):
        p = str(tmp_path / "broken.py")
        res = _make_parse(p, parseable=False, syntax_error="SyntaxError: bad")
        output = _build_C([res], str(tmp_path))
        assert "NO PASS" in output

    def test_no_python_files_message(self, tmp_path):
        output = _build_C([], str(tmp_path))
        assert "no Python files found" in output


# ---------------------------------------------------------------------------
# _build_D — ghost import tensions
# ---------------------------------------------------------------------------

class TestBuildDGhostImports:
    def test_ghost_import_tension_included(self, tmp_path):
        p = str(tmp_path / "g.py")
        res = _make_parse(p, parseable=True, ghost_imports=["missing_lib"])
        inv = InventoryReport(files=[], duplicates=[], summary={})
        entries = []
        output = _build_D(inv, [res], entries)
        assert "GHOST-IMPORT" in output
        assert "missing_lib" in output

    def test_no_tensions_message(self, tmp_path):
        res = _make_parse(str(tmp_path / "clean.py"), parseable=True)
        inv = InventoryReport(files=[], duplicates=[], summary={})
        entries = []
        output = _build_D(inv, [res], entries)
        assert "No unresolved tensions" in output

    def test_duplicate_tension_included(self, tmp_path):
        a = str(tmp_path / "a.py")
        b = str(tmp_path / "b.py")
        inv = InventoryReport(files=[], duplicates=[(a, b)], summary={})
        output = _build_D(inv, [], [])
        assert "DUPLICATE" in output

    def test_syntax_error_tension_included(self, tmp_path):
        p = str(tmp_path / "broken.py")
        res = _make_parse(p, parseable=False, syntax_error="SyntaxError: oh no")
        inv = InventoryReport(files=[], duplicates=[], summary={})
        output = _build_D(inv, [res], [])
        assert "SYNTAX-ERROR" in output

    def test_scorecard_no_pass_tension_included(self, tmp_path):
        entry = ScorecardEntry("M1", "File completeness", "NO PASS", "none")
        inv = InventoryReport(files=[], duplicates=[], summary={})
        output = _build_D(inv, [], [entry])
        assert "SCORECARD-M1" in output


# ---------------------------------------------------------------------------
# _build_E — refactor (ghost_paths) and quarantine (UNKNOWN/OUTPUT)
# ---------------------------------------------------------------------------

def _make_record(path, file_type=CODE_PYTHON, sha256="abc", dup_of=None):
    return FileRecord(path=path, file_type=file_type,
                      size_bytes=100, sha256=sha256, is_duplicate_of=dup_of)


class TestBuildEGhostAndUnknown:
    def test_ghost_import_file_goes_to_refactor(self, tmp_path):
        p = str(tmp_path / "ghost.py")
        rec = _make_record(p, CODE_PYTHON)
        inv = InventoryReport(files=[rec], duplicates=[], summary={CODE_PYTHON: 1})
        res = _make_parse(p, parseable=True, ghost_imports=["fake_mod"])
        output = _build_E(inv, [res], str(tmp_path))
        assert "REFACTOR" in output
        assert "ghost.py" in output

    def test_unknown_file_goes_to_quarantine(self, tmp_path):
        p = str(tmp_path / "weird.xyz")
        rec = _make_record(p, UNKNOWN, sha256="def")
        inv = InventoryReport(files=[rec], duplicates=[], summary={UNKNOWN: 1})
        output = _build_E(inv, [], str(tmp_path))
        assert "QUARANTINE" in output
        assert "weird.xyz" in output

    def test_output_file_goes_to_quarantine(self, tmp_path):
        p = str(tmp_path / "plot.png")
        rec = _make_record(p, OUTPUT, sha256="ghi")
        inv = InventoryReport(files=[rec], duplicates=[], summary={OUTPUT: 1})
        output = _build_E(inv, [], str(tmp_path))
        assert "QUARANTINE" in output

    def test_parseable_clean_file_goes_to_keep(self, tmp_path):
        p = str(tmp_path / "good.py")
        rec = _make_record(p, CODE_PYTHON)
        inv = InventoryReport(files=[rec], duplicates=[], summary={CODE_PYTHON: 1})
        res = _make_parse(p, parseable=True, ghost_imports=[])
        output = _build_E(inv, [res], str(tmp_path))
        assert "KEEP" in output
        assert "good.py" in output

    def test_duplicate_file_goes_to_throw(self, tmp_path):
        a = str(tmp_path / "a.py")
        b = str(tmp_path / "b.py")
        reca = _make_record(a, CODE_PYTHON, sha256="same")
        recb = _make_record(b, CODE_PYTHON, sha256="same", dup_of=a)
        inv = InventoryReport(files=[reca, recb], duplicates=[(a, b)], summary={CODE_PYTHON: 2})
        output = _build_E(inv, [], str(tmp_path))
        assert "THROW" in output

    def test_broken_py_goes_to_quarantine(self, tmp_path):
        p = str(tmp_path / "broken.py")
        rec = _make_record(p, CODE_PYTHON, sha256="zzz")
        inv = InventoryReport(files=[rec], duplicates=[], summary={CODE_PYTHON: 1})
        res = _make_parse(p, parseable=False, syntax_error="err")
        output = _build_E(inv, [res], str(tmp_path))
        assert "QUARANTINE" in output


# ---------------------------------------------------------------------------
# Integration — ColdPass on a directory with a ghost-import Python file
# ---------------------------------------------------------------------------

class TestColdPassGhostIntegration:
    def test_ghost_import_appears_in_tensions(self, tmp_path):
        p = tmp_path / "ghosty.py"
        p.write_text(
            "import totally_nonexistent_pkg_xyz_abc\n\ndef foo(): pass\n"
        )
        report = ColdPass().run(str(tmp_path))
        assert "GHOST-IMPORT" in report.tensions

    def test_ghost_import_partial_pass_in_file_audit(self, tmp_path):
        p = tmp_path / "ghosty.py"
        p.write_text(
            "import totally_nonexistent_pkg_xyz_abc\n\ndef foo(): pass\n"
        )
        report = ColdPass().run(str(tmp_path))
        assert "PARTIAL PASS" in report.file_audit

    def test_ghost_import_file_in_refactor_section(self, tmp_path):
        p = tmp_path / "ghosty.py"
        p.write_text(
            "import totally_nonexistent_pkg_xyz_abc\n\ndef foo(): pass\n"
        )
        report = ColdPass().run(str(tmp_path))
        assert "REFACTOR" in report.cleanup_plan

    def test_unknown_file_in_quarantine_section(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("some raw data\n")
        report = ColdPass().run(str(tmp_path))
        assert "QUARANTINE" in report.cleanup_plan
