"""File inventory, hashing, and manifest building.

Builds a structured manifest of a directory tree with SHA-256 hashes and file type
classification. Safe ZIP extraction with path-traversal protection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_EXT_KIND: Dict[str, str] = {
    ".json": "json", ".jsonl": "jsonl",
    ".py": "python", ".pyi": "python",
    ".md": "markdown", ".txt": "text",
    ".csv": "csv", ".tsv": "csv",
    ".pdf": "pdf", ".html": "html", ".htm": "html",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".svg": "image", ".webp": "image",
    ".zip": "archive", ".tar": "archive", ".gz": "archive",
    ".yaml": "config", ".yml": "config", ".toml": "config",
    ".js": "javascript", ".ts": "typescript",
    ".sh": "shell",
}


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA-256 of a file. Returns None if file does not exist."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def kind_for(path: Path) -> str:
    """Classify a file by extension."""
    return _EXT_KIND.get(path.suffix.lower(), "unknown")


def build_manifest(root: Path) -> List[Dict[str, Any]]:
    """Recursively build a manifest of all files under root."""
    entries: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
            entries.append({
                "relpath": str(p.relative_to(root)),
                "name": p.name,
                "kind": kind_for(p),
                "ext": p.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(p),
            })
        except OSError:
            pass
    return entries


def aggregate_hash(entries: List[Dict[str, Any]]) -> str:
    """Compute a deterministic aggregate hash over a manifest entry list."""
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(
    root: Path,
    out_dir: Path,
    label: str = "manifest",
) -> Dict[str, Any]:
    """Build and write manifest as both JSON and CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = build_manifest(root)
    agg = aggregate_hash(entries)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "generated_utc": now,
        "root": str(root),
        "file_count": len(entries),
        "aggregate_hash": agg,
        "production_status": "LOCKED",
        "files": entries,
    }
    json_path = out_dir / f"{label}.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / f"{label}.csv"
    if entries:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            w.writeheader()
            w.writerows(entries)
    return manifest


def safe_extract_zip(zip_path: Path, dest: Path) -> List[str]:
    """Extract ZIP to dest with path-traversal protection. Returns list of extracted paths."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Reject absolute paths and .. traversals
            name = member.filename
            if name.startswith("/") or ".." in name:
                continue
            target = dest / name
            if not str(target.resolve()).startswith(str(dest.resolve())):
                continue
            z.extract(member, dest)
            extracted.append(str(target))
    return extracted
