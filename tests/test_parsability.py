"""Tests for omniagis.audit.parsability — ParsabilityChecker and helpers."""
from __future__ import annotations

import os
import textwrap

import pytest

from omniagis.audit.parsability import (
    ParseResult,
    ParsabilityChecker,
    _extract_imports,
    _has_definitions,
    _is_available,
)
import ast


# ---------------------------------------------------------------------------
# _extract_imports
# ---------------------------------------------------------------------------

class TestExtractImports:
    def _parse(self, src: str):
        return ast.parse(textwrap.dedent(src))

    def test_bare_import(self):
        tree = self._parse("import os")
        assert _extract_imports(tree) == ["os"]

    def test_from_import(self):
        tree = self._parse("from pathlib import Path")
        assert _extract_imports(tree) == ["pathlib"]

    def test_relative_import_excluded(self):
        tree = self._parse("from . import sibling")
        assert _extract_imports(tree) == []

    def test_relative_dotted_import_excluded(self):
        tree = self._parse("from .utils import helper")
        assert _extract_imports(tree) == []

    def test_multiple_imports(self):
        tree = self._parse("import os\nimport sys")
        assert set(_extract_imports(tree)) == {"os", "sys"}

    def test_aliased_import(self):
        tree = self._parse("import numpy as np")
        assert _extract_imports(tree) == ["numpy"]

    def test_dotted_from_import(self):
        tree = self._parse("from os.path import join")
        assert _extract_imports(tree) == ["os.path"]


# ---------------------------------------------------------------------------
# _has_definitions
# ---------------------------------------------------------------------------

class TestHasDefinitions:
    def _parse(self, src: str):
        return ast.parse(textwrap.dedent(src))

    def test_function_def_detected(self):
        tree = self._parse("def foo(): pass")
        assert _has_definitions(tree) is True

    def test_class_def_detected(self):
        tree = self._parse("class Foo: pass")
        assert _has_definitions(tree) is True

    def test_async_function_detected(self):
        tree = self._parse("async def foo(): pass")
        assert _has_definitions(tree) is True

    def test_no_defs_in_empty_module(self):
        tree = self._parse("")
        assert _has_definitions(tree) is False

    def test_no_defs_in_assignment_only(self):
        tree = self._parse("x = 1\ny = 2")
        assert _has_definitions(tree) is False

    def test_no_defs_in_import_only(self):
        tree = self._parse("import os")
        assert _has_definitions(tree) is False


# ---------------------------------------------------------------------------
# _is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_stdlib_os_available(self):
        assert _is_available("os") is True

    def test_stdlib_sys_available(self):
        assert _is_available("sys") is True

    def test_dotted_stdlib_available(self):
        assert _is_available("os.path") is True

    def test_nonexistent_module_not_available(self):
        assert _is_available("totally_fake_module_xyz_123") is False


# ---------------------------------------------------------------------------
# ParsabilityChecker.check_file
# ---------------------------------------------------------------------------

class TestCheckFile:
    def test_valid_python_file(self, tmp_path):
        f = tmp_path / "good.py"
        f.write_text("def foo():\n    return 1\n")
        result = ParsabilityChecker().check_file(str(f))
        assert result.parseable is True
        assert result.syntax_error is None

    def test_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(:\n    pass\n")
        result = ParsabilityChecker().check_file(str(f))
        assert result.parseable is False
        assert result.syntax_error is not None
        assert "SyntaxError" in result.syntax_error

    def test_missing_file_returns_unparseable(self, tmp_path):
        result = ParsabilityChecker().check_file(str(tmp_path / "nonexistent.py"))
        assert result.parseable is False
        assert result.syntax_error is not None

    def test_imports_extracted(self, tmp_path):
        f = tmp_path / "imports.py"
        f.write_text("import os\nimport sys\n")
        result = ParsabilityChecker().check_file(str(f))
        assert "os" in result.imports
        assert "sys" in result.imports

    def test_ghost_import_detected(self, tmp_path):
        f = tmp_path / "ghost.py"
        f.write_text("import totally_fake_module_xyz_123\n")
        result = ParsabilityChecker().check_file(str(f))
        assert "totally_fake_module_xyz_123" in result.ghost_imports

    def test_no_ghost_for_stdlib(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("import os\nimport sys\n")
        result = ParsabilityChecker().check_file(str(f))
        assert result.ghost_imports == []

    def test_relative_imports_not_ghost(self, tmp_path):
        f = tmp_path / "rel.py"
        f.write_text("from . import sibling\nfrom .utils import helper\n")
        result = ParsabilityChecker().check_file(str(f))
        assert result.ghost_imports == []

    def test_path_preserved_in_result(self, tmp_path):
        f = tmp_path / "myfile.py"
        f.write_text("x = 1\n")
        result = ParsabilityChecker().check_file(str(f))
        assert result.path == str(f)


# ---------------------------------------------------------------------------
# ParsabilityChecker.check_directory
# ---------------------------------------------------------------------------

class TestCheckDirectory:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        results = ParsabilityChecker().check_directory(str(tmp_path))
        paths = [r.path for r in results]
        assert any("a.py" in p for p in paths)
        assert any("b.py" in p for p in paths)

    def test_skips_non_py_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# hi\n")
        (tmp_path / "data.csv").write_text("a,b\n")
        results = ParsabilityChecker().check_directory(str(tmp_path))
        assert results == []

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.py").write_text("x = 1\n")
        results = ParsabilityChecker().check_directory(str(tmp_path))
        assert results == []

    def test_recurses_into_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("z = 3\n")
        results = ParsabilityChecker().check_directory(str(tmp_path))
        assert any("deep.py" in r.path for r in results)

    def test_empty_dir_returns_empty(self, tmp_path):
        assert ParsabilityChecker().check_directory(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# ParsabilityChecker.is_pseudo_code
# ---------------------------------------------------------------------------

class TestIsPseudoCode:
    def test_file_with_only_comments_is_pseudo(self, tmp_path):
        f = tmp_path / "pseudo.py"
        f.write_text("# just a comment\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is True

    def test_file_with_function_is_not_pseudo(self, tmp_path):
        f = tmp_path / "real.py"
        f.write_text("def foo():\n    pass\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is False

    def test_file_with_class_is_not_pseudo(self, tmp_path):
        f = tmp_path / "cls.py"
        f.write_text("class Bar:\n    pass\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is False

    def test_init_py_never_pseudo(self, tmp_path):
        f = tmp_path / "__init__.py"
        f.write_text("# empty init\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is False

    def test_syntax_error_file_returns_false(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def foo(:\n    pass\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is False

    def test_missing_file_returns_false(self, tmp_path):
        assert ParsabilityChecker().is_pseudo_code(str(tmp_path / "ghost.py")) is False

    def test_assignment_only_is_pseudo(self, tmp_path):
        f = tmp_path / "assign.py"
        f.write_text("x = 1\ny = 2\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is True

    def test_import_only_is_pseudo(self, tmp_path):
        f = tmp_path / "imports_only.py"
        f.write_text("import os\nimport sys\n")
        assert ParsabilityChecker().is_pseudo_code(str(f)) is True
