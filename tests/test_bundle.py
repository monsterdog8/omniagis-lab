"""Tests for the BundleAuditor — bundle manifest hash + chain validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from omniagis.audit.bundle import (
    FAIL_CLOSED,
    PARTIAL_PASS,
    PASS,
    REASON_ARTIFACT_MISSING,
    REASON_CHAIN_HASH_MISMATCH,
    REASON_CHAIN_NO_PREV,
    REASON_CHAIN_OK,
    REASON_CHAIN_PREV_MISSING,
    REASON_CHAIN_SKIPPED,
    REASON_HASH_INVALID,
    REASON_HASH_MISSING,
    REASON_HASH_SKIPPED,
    REASON_HASH_VALID,
    BundleAuditor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_manifest(directory: Path, data: dict) -> Path:
    p = directory / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_artifact(directory: Path, name: str, content: bytes = b"hello") -> Path:
    p = directory / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_basic_load(self, tmp_path):
        manifest_path = _write_manifest(tmp_path, {
            "version": "5.1",
            "name": "test-bundle",
            "description": "Test",
            "artifacts": [
                {"name": "a1", "path": "file.json"}
            ],
        })
        m = BundleAuditor().load_manifest(str(manifest_path))
        assert m.version == "5.1"
        assert m.name == "test-bundle"
        assert len(m.artifacts) == 1
        assert m.artifacts[0].name == "a1"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BundleAuditor().load_manifest(str(tmp_path / "nonexistent.json"))


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------

class TestPresenceCheck:
    def test_artifact_present(self, tmp_path):
        content = b'{"key": "value"}'
        art = _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "a.json"}],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.results[0].present is True

    def test_artifact_missing(self, tmp_path):
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "missing.json"}],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.present is False
        assert r.hash_reason == REASON_HASH_SKIPPED
        assert r.chain_reason == REASON_CHAIN_SKIPPED
        assert r.verdict == FAIL_CLOSED

    def test_global_verdict_fail_closed_when_artifact_missing(self, tmp_path):
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "missing.json"}],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.global_verdict == FAIL_CLOSED
        assert report.n_missing == 1
        assert REASON_ARTIFACT_MISSING in report.fail_reasons


# ---------------------------------------------------------------------------
# Hash checks
# ---------------------------------------------------------------------------

class TestHashCheck:
    def test_hash_valid(self, tmp_path):
        content = b"data content"
        art = _write_artifact(tmp_path, "f.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "f.json", "sha256": _sha256(content)}],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.hash_reason == REASON_HASH_VALID
        assert r.actual_sha256 == _sha256(content)

    def test_hash_invalid(self, tmp_path):
        content = b"data content"
        art = _write_artifact(tmp_path, "f.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "f.json", "sha256": "deadbeef" * 8}],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.hash_reason == REASON_HASH_INVALID
        assert r.verdict == FAIL_CLOSED

    def test_hash_missing_is_partial_pass(self, tmp_path):
        art = _write_artifact(tmp_path, "f.json", b"data")
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "f.json"}],  # no sha256
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.hash_reason == REASON_HASH_MISSING
        # present + no declared hash + no declared prev → PARTIAL PASS (not FAIL)
        assert r.verdict == PARTIAL_PASS

    def test_global_verdict_fail_closed_on_hash_mismatch(self, tmp_path):
        content = b"data"
        _write_artifact(tmp_path, "f.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "f.json", "sha256": "0" * 64}],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.global_verdict == FAIL_CLOSED


# ---------------------------------------------------------------------------
# Chain checks
# ---------------------------------------------------------------------------

class TestChainCheck:
    def test_no_prev_is_chain_no_prev(self, tmp_path):
        _write_artifact(tmp_path, "f.json", b"x")
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "a1", "path": "f.json", "sha256": _sha256(b"x")}],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.results[0].chain_reason == REASON_CHAIN_NO_PREV

    def test_chain_ok_when_prev_present_and_hash_matches(self, tmp_path):
        prev_content = b"prev data"
        curr_content = b"curr data"
        _write_artifact(tmp_path, "prev.json", prev_content)
        _write_artifact(tmp_path, "curr.json", curr_content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{
                "name": "a2",
                "path": "curr.json",
                "sha256": _sha256(curr_content),
                "prev_artifact_path": "prev.json",
                "prev_artifact_sha256": _sha256(prev_content),
            }],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.chain_reason == REASON_CHAIN_OK
        assert r.verdict == PASS

    def test_chain_prev_missing(self, tmp_path):
        curr_content = b"curr data"
        _write_artifact(tmp_path, "curr.json", curr_content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{
                "name": "a2",
                "path": "curr.json",
                "sha256": _sha256(curr_content),
                "prev_artifact_path": "MISSING_PREV.json",
                "prev_artifact_sha256": "aabbcc" * 10 + "aabb",
            }],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.chain_reason == REASON_CHAIN_PREV_MISSING
        assert r.verdict == FAIL_CLOSED
        assert report.global_verdict == FAIL_CLOSED
        assert REASON_CHAIN_PREV_MISSING in report.fail_reasons

    def test_chain_hash_mismatch(self, tmp_path):
        prev_content = b"prev data"
        curr_content = b"curr data"
        _write_artifact(tmp_path, "prev.json", prev_content)
        _write_artifact(tmp_path, "curr.json", curr_content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{
                "name": "a2",
                "path": "curr.json",
                "sha256": _sha256(curr_content),
                "prev_artifact_path": "prev.json",
                "prev_artifact_sha256": "00" * 32,  # wrong hash
            }],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.chain_reason == REASON_CHAIN_HASH_MISMATCH
        assert r.verdict == FAIL_CLOSED

    def test_chain_ok_without_prev_sha256(self, tmp_path):
        """Chain link with prev_artifact_path but no sha256 → CHAIN_OK (presence only)."""
        prev_content = b"prev"
        curr_content = b"curr"
        _write_artifact(tmp_path, "prev.json", prev_content)
        _write_artifact(tmp_path, "curr.json", curr_content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{
                "name": "a2",
                "path": "curr.json",
                "sha256": _sha256(curr_content),
                "prev_artifact_path": "prev.json",
                # no prev_artifact_sha256 declared
            }],
        })
        report = BundleAuditor().audit(str(mpath))
        r = report.results[0]
        assert r.chain_reason == REASON_CHAIN_OK


# ---------------------------------------------------------------------------
# Multi-artifact bundles
# ---------------------------------------------------------------------------

class TestMultiArtifact:
    def test_all_pass(self, tmp_path):
        c1, c2 = b"one", b"two"
        _write_artifact(tmp_path, "a.json", c1)
        _write_artifact(tmp_path, "b.json", c2)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "multi", "description": "",
            "artifacts": [
                {"name": "a1", "path": "a.json", "sha256": _sha256(c1)},
                {"name": "a2", "path": "b.json", "sha256": _sha256(c2)},
            ],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.global_verdict == PASS
        assert report.n_declared == 2
        assert report.n_present == 2
        assert report.n_missing == 0
        assert report.n_hash_valid == 2

    def test_one_missing_fails_closed(self, tmp_path):
        c1 = b"one"
        _write_artifact(tmp_path, "a.json", c1)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "multi", "description": "",
            "artifacts": [
                {"name": "a1", "path": "a.json", "sha256": _sha256(c1)},
                {"name": "a2", "path": "MISSING.json"},
            ],
        })
        report = BundleAuditor().audit(str(mpath))
        assert report.global_verdict == FAIL_CLOSED
        assert report.n_missing == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_render_contains_global_verdict(self, tmp_path):
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [],
        })
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        text = auditor.render(report)
        assert "GLOBAL VERDICT" in text
        assert report.global_verdict in text

    def test_render_lists_missing_artifact(self, tmp_path):
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "b", "description": "",
            "artifacts": [{"name": "ghost", "path": "not_here.json"}],
        })
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        text = auditor.render(report)
        assert "MISSING" in text
        assert "ghost" in text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicApi:
    def test_bundle_auditor_importable_from_top_level(self):
        from omniagis import BundleAuditor as BA
        assert BA is not None

    def test_bundle_auditor_importable_from_audit(self):
        from omniagis.audit import BundleAuditor as BA
        assert BA is not None
