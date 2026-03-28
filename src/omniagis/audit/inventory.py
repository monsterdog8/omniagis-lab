"""File inventory and type classification for OMNIÆGIS audit."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# File type constants
CODE_PYTHON = "CODE_PYTHON"
CODE_OTHER = "CODE_OTHER"
SPEC = "SPEC"
DOC = "DOC"
DATA = "DATA"
PDF = "PDF"
OUTPUT = "OUTPUT"
UNKNOWN = "UNKNOWN"

_EXT_MAP: Dict[str, str] = {
    ".py": CODE_PYTHON,
    ".pyx": CODE_PYTHON,
    ".pyi": CODE_PYTHON,
    # other code
    ".c": CODE_OTHER,
    ".cpp": CODE_OTHER,
    ".h": CODE_OTHER,
    ".hpp": CODE_OTHER,
    ".js": CODE_OTHER,
    ".ts": CODE_OTHER,
    ".rb": CODE_OTHER,
    ".java": CODE_OTHER,
    ".rs": CODE_OTHER,
    ".go": CODE_OTHER,
    ".sh": CODE_OTHER,
    ".bash": CODE_OTHER,
    ".r": CODE_OTHER,
    ".jl": CODE_OTHER,
    ".m": CODE_OTHER,
    # specs / config
    ".toml": SPEC,
    ".cfg": SPEC,
    ".ini": SPEC,
    ".yaml": SPEC,
    ".yml": SPEC,
    ".json": SPEC,
    ".lock": SPEC,
    # docs
    ".md": DOC,
    ".rst": DOC,
    ".txt": DOC,
    ".tex": DOC,
    ".ipynb": DOC,
    # data
    ".csv": DATA,
    ".tsv": DATA,
    ".npy": DATA,
    ".npz": DATA,
    ".h5": DATA,
    ".hdf5": DATA,
    ".parquet": DATA,
    ".feather": DATA,
    ".pkl": DATA,
    ".pickle": DATA,
    ".mat": DATA,
    # pdf
    ".pdf": PDF,
    # outputs / build artefacts
    ".log": OUTPUT,
    ".out": OUTPUT,
    ".png": OUTPUT,
    ".jpg": OUTPUT,
    ".jpeg": OUTPUT,
    ".svg": OUTPUT,
    ".gif": OUTPUT,
    ".html": OUTPUT,
    ".htm": OUTPUT,
}

_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", ".venv", "venv", ".eggs"}


@dataclass
class FileRecord:
    """Metadata for a single file in the inventory."""

    path: str
    file_type: str
    size_bytes: int
    sha256: str
    is_duplicate_of: Optional[str] = None


@dataclass
class InventoryReport:
    """Full inventory of a directory tree."""

    files: List[FileRecord] = field(default_factory=list)
    duplicates: List[Tuple[str, str]] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)


def _classify_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _EXT_MAP.get(ext, UNKNOWN)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FileInventory:
    """Walk a directory tree and build a structured inventory."""

    def build(self, root: str) -> InventoryReport:
        """Recursively inventory *root*.

        Parameters
        ----------
        root:
            Absolute path to the directory to scan.

        Returns
        -------
        InventoryReport
        """
        seen_hashes: Dict[str, str] = {}  # hash → first path
        records: List[FileRecord] = []
        duplicates: List[Tuple[str, str]] = []
        summary: Dict[str, int] = {}

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune directories we never want to recurse into
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                    digest = _sha256(fpath)
                    ftype = _classify_file(fpath)
                except OSError:
                    continue

                is_dup_of: Optional[str] = None
                if digest in seen_hashes:
                    first = seen_hashes[digest]
                    is_dup_of = first
                    duplicates.append((first, fpath))
                else:
                    seen_hashes[digest] = fpath

                records.append(
                    FileRecord(
                        path=fpath,
                        file_type=ftype,
                        size_bytes=size,
                        sha256=digest,
                        is_duplicate_of=is_dup_of,
                    )
                )
                summary[ftype] = summary.get(ftype, 0) + 1

        return InventoryReport(files=records, duplicates=duplicates, summary=summary)
