"""Tests for MODULE_007_MANIFEST — extracted from tests/test_gpts_core.py::TestManifest."""
from __future__ import annotations

import pathlib
import zipfile

import pytest

from gpts_core.manifest import (
    kind_for,
    sha256_file,
    build_manifest,
    aggregate_hash,
    write_manifest,
    safe_extract_zip,
)


class TestManifest:
    def test_kind_for_known_extensions(self):
        assert kind_for(pathlib.Path("file.py")) == "python"
        assert kind_for(pathlib.Path("data.csv")) == "csv"
        assert kind_for(pathlib.Path("archive.zip")) == "archive"

    def test_kind_for_unknown(self):
        assert kind_for(pathlib.Path("file.xyz")) == "unknown"

    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"deterministic content")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_sha256_file_nonexistent(self, tmp_path):
        assert sha256_file(tmp_path / "nope.txt") is None

    def test_build_manifest(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.csv").write_text("a,b\n1,2")
        entries = build_manifest(tmp_path)
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "a.py" in names

    def test_aggregate_hash_deterministic(self, tmp_path):
        (tmp_path / "x.txt").write_bytes(b"data")
        entries = build_manifest(tmp_path)
        h1 = aggregate_hash(entries)
        h2 = aggregate_hash(entries)
        assert h1 == h2

    def test_write_manifest(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").write_text("hello")
        out = tmp_path / "out"
        result = write_manifest(src, out, label="test_manifest")
        assert result["file_count"] == 1
        assert "aggregate_hash" in result
        assert (out / "test_manifest.json").exists()

    def test_safe_extract_zip(self, tmp_path):
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("normal/file.txt", "content")
        extracted = safe_extract_zip(zp, tmp_path / "dest")
        assert len(extracted) == 1

    def test_safe_extract_zip_blocks_traversal(self, tmp_path):
        zp = tmp_path / "evil.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("../evil.txt", "bad")
        extracted = safe_extract_zip(zp, tmp_path / "dest")
        assert all("evil.txt" not in e for e in extracted)
