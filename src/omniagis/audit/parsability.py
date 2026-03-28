"""Python parsability and import analysis for OMNIÆGIS audit."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParseResult:
    """Result of parsing a single Python file."""

    path: str
    parseable: bool
    syntax_error: Optional[str]
    imports: List[str] = field(default_factory=list)
    ghost_imports: List[str] = field(default_factory=list)


def _extract_imports(tree: ast.Module) -> List[str]:
    """Return absolute module names for all non-relative imports in *tree*."""
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 means a relative import (from . import x / from .mod import x)
            # Relative imports are intra-package — skip them for ghost-import analysis.
            if node.level == 0 and node.module:
                names.append(node.module)
    return names


def _is_available(module_name: str) -> bool:
    """Return True if *module_name* is importable (stdlib or installed)."""
    top = module_name.split(".")[0]
    if top in sys.stdlib_module_names:  # type: ignore[attr-defined]  # py3.10+
        return True
    if top in sys.modules:
        return True
    try:
        spec = importlib.util.find_spec(top)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _has_definitions(tree: ast.Module) -> bool:
    """Return True if the module contains at least one function or class def."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
    return False


class ParsabilityChecker:
    """Check Python files for syntax validity and resolvable imports."""

    def check_file(self, path: str) -> ParseResult:
        """Parse a single Python file and analyse its imports.

        Parameters
        ----------
        path:
            Absolute path to a ``.py`` file.

        Returns
        -------
        ParseResult
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError as exc:
            return ParseResult(
                path=path,
                parseable=False,
                syntax_error=str(exc),
            )

        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            return ParseResult(
                path=path,
                parseable=False,
                syntax_error=f"{exc.__class__.__name__}: {exc}",
            )

        imports = _extract_imports(tree)
        ghost_imports = [name for name in imports if not _is_available(name)]

        return ParseResult(
            path=path,
            parseable=True,
            syntax_error=None,
            imports=imports,
            ghost_imports=ghost_imports,
        )

    def check_directory(self, path: str) -> List[ParseResult]:
        """Recursively parse all ``.py`` files under *path*.

        Parameters
        ----------
        path:
            Root directory to search.

        Returns
        -------
        list of ParseResult
        """
        results: List[ParseResult] = []
        skip_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", ".venv", "venv"}
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [
                d for d in dirnames
                if d not in skip_dirs and not d.endswith(".egg-info")
            ]
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    results.append(self.check_file(os.path.join(dirpath, fname)))
        return results

    def is_pseudo_code(self, path: str) -> bool:
        """Return True if the file parses but defines no functions or classes.

        A file of only comments, docstrings, ``pass`` statements, or string
        literals is considered pseudo-code.

        ``__init__.py`` files are excluded from this check: they legitimately
        contain only imports and re-exports and must not be classified as
        pseudo-code.
        """
        if os.path.basename(path) == "__init__.py":
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            return False

        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return False

        return not _has_definitions(tree)
