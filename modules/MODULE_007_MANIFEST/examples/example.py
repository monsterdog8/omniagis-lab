"""MODULE_007_MANIFEST — runnable usage examples.

Demonstrates: sha256_file, sha256_bytes, kind_for, build_manifest,
aggregate_hash, write_manifest, and safe_extract_zip.
"""
from __future__ import annotations

import pathlib
import tempfile
import zipfile

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
# Example 1: Hash a file and classify it by type
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    # Create a sample Python file
    py_file = tmp_path / "model.py"
    py_file.write_text("def predict(x):\n    return x * 2\n")

    # Compute SHA-256
    digest = sha256_file(py_file)
    print(f"[Example 1] sha256_file: {digest}")

    # Classify by extension
    file_kind = kind_for(py_file)
    print(f"[Example 1] kind_for(.py): {file_kind}")  # -> python

    # Hash bytes directly
    raw_hash = sha256_bytes(b"hello, manifest")
    print(f"[Example 1] sha256_bytes: {raw_hash}")

    # Missing file returns None
    missing = sha256_file(tmp_path / "does_not_exist.txt")
    print(f"[Example 1] sha256_file (missing): {missing}")  # -> None

# ---------------------------------------------------------------------------
# Example 2: Build a directory manifest and compute aggregate hash
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    # Populate a small project directory
    (tmp_path / "README.md").write_text("# Project\nDescription here.\n")
    (tmp_path / "config.yaml").write_text("version: 1\nmode: lab\n")
    sub = tmp_path / "data"
    sub.mkdir()
    (sub / "samples.csv").write_text("id,value\n1,0.8\n2,0.6\n")

    entries = build_manifest(tmp_path)
    print(f"\n[Example 2] build_manifest: {len(entries)} files")
    for e in entries:
        print(f"  {e['relpath']!r:30s}  kind={e['kind']!r:10s}  size={e['size_bytes']} bytes")

    agg = aggregate_hash(entries)
    print(f"[Example 2] aggregate_hash: {agg}")

# ---------------------------------------------------------------------------
# Example 3: Write manifest to disk and safely extract a ZIP
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)

    # Source tree
    src = tmp_path / "src"
    src.mkdir()
    (src / "run.py").write_text("print('run')\n")
    (src / "notes.txt").write_text("Lab notes.\n")

    # Write manifest (JSON + CSV) to out/
    out = tmp_path / "out"
    result = write_manifest(src, out, label="lab_manifest")
    print(f"\n[Example 3] write_manifest: file_count={result['file_count']}")
    print(f"[Example 3] aggregate_hash: {result['aggregate_hash']}")
    print(f"[Example 3] JSON written: {(out / 'lab_manifest.json').exists()}")
    print(f"[Example 3] CSV written:  {(out / 'lab_manifest.csv').exists()}")

    # Create a ZIP and extract safely (path-traversal blocked)
    zp = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("subdir/data.csv", "a,b\n1,2\n")
        z.writestr("../evil.txt", "this should be blocked")  # traversal attempt

    dest = tmp_path / "extracted"
    extracted = safe_extract_zip(zp, dest)
    print(f"[Example 3] safe_extract_zip: {len(extracted)} file(s) extracted (traversal blocked)")
    for path in extracted:
        print(f"  extracted: {path}")
