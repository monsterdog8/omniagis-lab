"""Tests for gpts_core.manifest — file inventory, hashing, manifest building."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from gpts_core.manifest import (
    sha256_file,
    sha256_bytes,
    kind_for,
    build_manifest,
    aggregate_hash,
    write_manifest,
    safe_extract_zip,
)


# ---------------------------------------------------------------------------
# sha256_file / sha256_bytes
# ---------------------------------------------------------------------------

class TestSha256:
    def test_sha256_file_existing(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_bytes(b"hello")
        h = sha256_file(f)
        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 71

    def test_sha256_file_missing(self, tmp_path):
        assert sha256_file(tmp_path / "missing.txt") is None

    def test_sha256_bytes_prefix(self):
        h = sha256_bytes(b"data")
        assert h.startswith("sha256:")
        assert len(h) == 71

    def test_sha256_bytes_deterministic(self):
        assert sha256_bytes(b"abc") == sha256_bytes(b"abc")


# ---------------------------------------------------------------------------
# kind_for
# ---------------------------------------------------------------------------

class TestKindFor:
    @pytest.mark.parametrize("name,expected", [
        ("script.py", "python"),
        ("data.json", "json"),
        ("data.jsonl", "jsonl"),
        ("readme.md", "markdown"),
        ("notes.txt", "text"),
        ("table.csv", "csv"),
        ("doc.pdf", "pdf"),
        ("page.html", "html"),
        ("img.png", "image"),
        ("img.jpg", "image"),
        ("archive.zip", "archive"),
        ("config.yaml", "config"),
        ("config.toml", "config"),
        ("app.js", "javascript"),
        ("types.ts", "typescript"),
        ("run.sh", "shell"),
        ("unknown.xyz", "unknown"),
    ])
    def test_kind(self, name, expected):
        assert kind_for(Path(name)) == expected


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_basic_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1", encoding="utf-8")
        (tmp_path / "b.json").write_text('{"k":1}', encoding="utf-8")
        entries = build_manifest(tmp_path)
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "a.py" in names
        assert "b.json" in names

    def test_entries_have_required_fields(self, tmp_path):
        (tmp_path / "f.py").write_text("pass", encoding="utf-8")
        entries = build_manifest(tmp_path)
        for key in ["relpath", "name", "kind", "ext", "size_bytes", "sha256"]:
            assert key in entries[0]

    def test_empty_directory(self, tmp_path):
        entries = build_manifest(tmp_path)
        assert entries == []

    def test_nested_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("pass", encoding="utf-8")
        entries = build_manifest(tmp_path)
        assert any("nested.py" in e["name"] for e in entries)

    def test_sha256_present_for_each_file(self, tmp_path):
        (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
        entries = build_manifest(tmp_path)
        assert entries[0]["sha256"] is not None
        assert entries[0]["sha256"].startswith("sha256:")


# ---------------------------------------------------------------------------
# aggregate_hash
# ---------------------------------------------------------------------------

class TestAggregateHash:
    def test_deterministic(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1", encoding="utf-8")
        entries = build_manifest(tmp_path)
        h1 = aggregate_hash(entries)
        h2 = aggregate_hash(entries)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1", encoding="utf-8")
        e1 = build_manifest(tmp_path)
        (tmp_path / "b.py").write_text("y=2", encoding="utf-8")
        e2 = build_manifest(tmp_path)
        assert aggregate_hash(e1) != aggregate_hash(e2)

    def test_empty_entries(self):
        h = aggregate_hash([])
        assert h.startswith("sha256:")


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_json_and_csv(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f.py").write_text("x=1", encoding="utf-8")
        out = tmp_path / "out"
        result = write_manifest(src, out, label="test")
        assert (out / "test.json").exists()
        assert (out / "test.csv").exists()

    def test_manifest_has_required_fields(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f.py").write_text("x=1", encoding="utf-8")
        out = tmp_path / "out"
        result = write_manifest(src, out)
        for key in ["generated_utc", "root", "file_count", "aggregate_hash",
                    "production_status", "files"]:
            assert key in result

    def test_production_status_locked(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        result = write_manifest(src, out)
        assert result["production_status"] == "LOCKED"


# ---------------------------------------------------------------------------
# safe_extract_zip
# ---------------------------------------------------------------------------

class TestSafeExtractZip:
    def _make_zip(self, tmp_path, name="archive.zip"):
        z = tmp_path / name
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")
        return z

    def test_extracts_normal_files(self, tmp_path):
        z = self._make_zip(tmp_path)
        dest = tmp_path / "extracted"
        extracted = safe_extract_zip(z, dest)
        assert any("file1.txt" in p for p in extracted)

    def test_rejects_absolute_path(self, tmp_path):
        z = tmp_path / "bad.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("/etc/passwd", "root:x:0:0")
        dest = tmp_path / "extracted"
        extracted = safe_extract_zip(z, dest)
        assert not any("passwd" in p for p in extracted)

    def test_rejects_path_traversal(self, tmp_path):
        z = tmp_path / "traversal.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../outside.txt", "evil")
        dest = tmp_path / "extracted"
        extracted = safe_extract_zip(z, dest)
        assert extracted == []
