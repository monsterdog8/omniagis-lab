# MODULE_007_MANIFEST

## Purpose

File inventory, SHA-256 hashing, manifest building, and safe ZIP extraction.

Builds a structured manifest of a directory tree — every file is hashed (SHA-256), classified by type, and recorded with its size. An aggregate deterministic hash covers the entire tree. Manifests are written as both JSON and CSV. ZIP extraction is path-traversal safe.

## Source

`gpts_core/manifest.py`

## Public API

| Function | Signature | Returns | Description |
|---|---|---|---|
| `sha256_file` | `(path: Path) -> Optional[str]` | `Optional[str]` | SHA-256 of a file; `None` if missing |
| `sha256_bytes` | `(data: bytes) -> str` | `str` | SHA-256 of raw bytes |
| `kind_for` | `(path: Path) -> str` | `str` | File type from extension |
| `build_manifest` | `(root: Path) -> List[Dict]` | `List[Dict[str, Any]]` | Full directory manifest |
| `aggregate_hash` | `(entries: List[Dict]) -> str` | `str` | Deterministic aggregate hash |
| `write_manifest` | `(root, out_dir, label) -> Dict` | `Dict[str, Any]` | Write JSON+CSV manifest |
| `safe_extract_zip` | `(zip_path, dest) -> List[str]` | `List[str]` | Traversal-safe ZIP extraction |

## Dependencies

All stdlib: `hashlib`, `zipfile`, `json`, `csv`, `pathlib`, `datetime`

## Coverage

90%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All outputs are local filesystem operations. No external validation is claimed.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_007.py` — extracted from `tests/test_gpts_core.py::TestManifest`
