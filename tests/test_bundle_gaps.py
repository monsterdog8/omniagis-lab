"""Coverage gaps for bundle.py — is_ok, PARTIAL PASS verdict, fail_reasons,
render() with failures, bundle.main() CLI (text + JSON + error paths)."""
from __future__ import annotations

import hashlib
import json
import sys

import pytest

from omniagis.audit.bundle import (
    FAIL_CLOSED,
    PARTIAL_PASS,
    PASS,
    REASON_CHAIN_HASH_MISMATCH,
    REASON_CHAIN_NO_PREV,
    REASON_CHAIN_OK,
    REASON_CHAIN_PREV_MISSING,
    REASON_CHAIN_SKIPPED,
    REASON_HASH_INVALID,
    REASON_HASH_MISSING,
    REASON_HASH_SKIPPED,
    REASON_HASH_VALID,
    ArtifactResult,
    ArtifactSpec,
    BundleAuditReport,
    BundleManifest,
    BundleAuditor,
    _resolve,
    main as bundle_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_manifest(directory, data: dict):
    p = directory / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_artifact(directory, name: str, content: bytes = b"hello"):
    p = directory / name
    p.write_bytes(content)
    return p


def _make_spec(name="art", path="/tmp/f.json", sha256=None,
               prev_path=None, prev_sha256=None) -> ArtifactSpec:
    return ArtifactSpec(name=name, path=path, sha256=sha256,
                        prev_artifact_path=prev_path,
                        prev_artifact_sha256=prev_sha256)


def _make_result(present=True, hash_reason=REASON_HASH_VALID,
                 chain_reason=REASON_CHAIN_NO_PREV,
                 actual_sha256="abc", spec=None) -> ArtifactResult:
    if spec is None:
        spec = _make_spec()
    return ArtifactResult(spec=spec, present=present,
                          hash_reason=hash_reason, chain_reason=chain_reason,
                          actual_sha256=actual_sha256 if present else None)


def _make_manifest(name="m") -> BundleManifest:
    return BundleManifest(version="5.1", name=name, description="test",
                          source_path="/tmp/m.json")


def _make_report(results) -> BundleAuditReport:
    return BundleAuditReport(manifest=_make_manifest(), results=results)


# ---------------------------------------------------------------------------
# ArtifactResult.is_ok
# ---------------------------------------------------------------------------

class TestArtifactResultIsOk:
    def test_true_when_present_hash_valid_chain_ok(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_OK)
        assert r.is_ok is True

    def test_true_when_chain_no_prev(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_NO_PREV)
        assert r.is_ok is True

    def test_false_when_not_present(self):
        r = _make_result(present=False, hash_reason=REASON_HASH_SKIPPED,
                         chain_reason=REASON_CHAIN_SKIPPED)
        assert r.is_ok is False

    def test_false_when_hash_invalid(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_INVALID,
                         chain_reason=REASON_CHAIN_NO_PREV)
        assert r.is_ok is False

    def test_false_when_hash_missing(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_MISSING,
                         chain_reason=REASON_CHAIN_NO_PREV)
        assert r.is_ok is False

    def test_false_when_chain_prev_missing(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_PREV_MISSING)
        assert r.is_ok is False


# ---------------------------------------------------------------------------
# BundleAuditReport.global_verdict — PARTIAL PASS path
# ---------------------------------------------------------------------------

class TestGlobalVerdictPartialPass:
    def test_partial_pass_when_hash_missing(self):
        # present, no chain declared (CHAIN_NO_PREV), hash not declared (HASH_MISSING)
        r = _make_result(present=True, hash_reason=REASON_HASH_MISSING,
                         chain_reason=REASON_CHAIN_NO_PREV)
        report = _make_report([r])
        assert report.global_verdict == PARTIAL_PASS

    def test_partial_pass_mixed_pass_and_partial(self):
        r1 = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                          chain_reason=REASON_CHAIN_NO_PREV)
        r2 = _make_result(present=True, hash_reason=REASON_HASH_MISSING,
                          chain_reason=REASON_CHAIN_NO_PREV)
        report = _make_report([r1, r2])
        assert report.global_verdict == PARTIAL_PASS


# ---------------------------------------------------------------------------
# BundleAuditReport.fail_reasons — hash invalid and chain mismatch paths
# ---------------------------------------------------------------------------

class TestFailReasons:
    def test_hash_invalid_included(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_INVALID,
                         chain_reason=REASON_CHAIN_NO_PREV)
        report = _make_report([r])
        assert REASON_HASH_INVALID in report.fail_reasons

    def test_chain_hash_mismatch_included(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_HASH_MISMATCH)
        report = _make_report([r])
        assert REASON_CHAIN_HASH_MISMATCH in report.fail_reasons

    def test_chain_prev_missing_included(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_PREV_MISSING)
        report = _make_report([r])
        assert REASON_CHAIN_PREV_MISSING in report.fail_reasons

    def test_deduplication_of_same_reason(self):
        r1 = _make_result(present=True, hash_reason=REASON_HASH_INVALID,
                          chain_reason=REASON_CHAIN_NO_PREV)
        r2 = _make_result(present=True, hash_reason=REASON_HASH_INVALID,
                          chain_reason=REASON_CHAIN_NO_PREV)
        report = _make_report([r1, r2])
        assert report.fail_reasons.count(REASON_HASH_INVALID) == 1

    def test_empty_when_all_ok(self):
        r = _make_result(present=True, hash_reason=REASON_HASH_VALID,
                         chain_reason=REASON_CHAIN_NO_PREV)
        report = _make_report([r])
        assert report.fail_reasons == []


# ---------------------------------------------------------------------------
# _resolve — relative path branch
# ---------------------------------------------------------------------------

class TestResolve:
    def test_relative_path_joined_to_manifest_dir(self):
        result = _resolve("artifacts/file.json", "/tmp/bundles")
        assert result == "/tmp/bundles/artifacts/file.json"

    def test_absolute_path_returned_unchanged(self):
        result = _resolve("/absolute/file.json", "/tmp/bundles")
        assert result == "/absolute/file.json"


# ---------------------------------------------------------------------------
# BundleAuditor.render() — hash_fails and chain_fails weakness sections
# ---------------------------------------------------------------------------

class TestRenderWeaknesses:
    def test_render_hash_invalid_weakness(self, tmp_path):
        content = b"some data"
        art = _write_artifact(tmp_path, "data.json", content)
        correct = _sha256(content)
        manifest_data = {
            "version": "5.1", "name": "test", "description": "",
            "artifacts": [{"name": "data", "path": "data.json", "sha256": "badhash"}],
        }
        mpath = _write_manifest(tmp_path, manifest_data)
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        rendered = auditor.render(report)
        assert "HASH_INVALID" in rendered
        assert "badhash" in rendered

    def test_render_chain_failure_weakness(self, tmp_path):
        content = b"main data"
        prev_content = b"previous data"
        art = _write_artifact(tmp_path, "main.json", content)
        prev = _write_artifact(tmp_path, "prev.json", prev_content)
        manifest_data = {
            "version": "5.1", "name": "test", "description": "",
            "artifacts": [{
                "name": "main",
                "path": "main.json",
                "sha256": _sha256(content),
                "prev_artifact_path": "prev.json",
                "prev_artifact_sha256": "wrong_prev_hash",
            }],
        }
        mpath = _write_manifest(tmp_path, manifest_data)
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        rendered = auditor.render(report)
        assert "CHAIN_HASH_MISMATCH" in rendered

    def test_render_no_weaknesses(self, tmp_path):
        content = b"clean data"
        _write_artifact(tmp_path, "clean.json", content)
        manifest_data = {
            "version": "5.1", "name": "test", "description": "",
            "artifacts": [{"name": "clean", "path": "clean.json",
                           "sha256": _sha256(content)}],
        }
        mpath = _write_manifest(tmp_path, manifest_data)
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        rendered = auditor.render(report)
        assert "None detected." in rendered

    def test_render_contains_global_verdict(self, tmp_path):
        content = b"data"
        _write_artifact(tmp_path, "f.json", content)
        manifest_data = {
            "version": "5.1", "name": "test", "description": "",
            "artifacts": [{"name": "f", "path": "f.json",
                           "sha256": _sha256(content)}],
        }
        mpath = _write_manifest(tmp_path, manifest_data)
        auditor = BundleAuditor()
        report = auditor.audit(str(mpath))
        rendered = auditor.render(report)
        assert "GLOBAL VERDICT" in rendered
        assert report.global_verdict in rendered


# ---------------------------------------------------------------------------
# bundle.main() — text output, JSON output, FileNotFoundError
# ---------------------------------------------------------------------------

class TestBundleMainText:
    def test_text_output_exits_with_code(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit) as exc:
            bundle_main([str(mpath)])
        assert exc.value.code in (0, 1, 2)

    def test_text_output_contains_verdict(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit):
            bundle_main([str(mpath)])
        out = capsys.readouterr().out
        assert "GLOBAL VERDICT" in out

    def test_fail_closed_exits_2(self, tmp_path):
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "missing", "path": "nonexistent.json",
                           "sha256": "abc123"}],
        })
        with pytest.raises(SystemExit) as exc:
            bundle_main([str(mpath)])
        assert exc.value.code == 2


class TestBundleMainJson:
    def test_json_output_is_valid_json(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit):
            bundle_main([str(mpath), "--output", "json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_json_has_required_keys(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit):
            bundle_main([str(mpath), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        for key in ("global_verdict", "artifacts", "n_declared", "n_present",
                    "n_missing", "fail_reasons", "bundle_name", "timestamp"):
            assert key in data

    def test_json_artifact_has_all_fields(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit):
            bundle_main([str(mpath), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        art = data["artifacts"][0]
        for field in ("name", "path", "present", "sha256_declared",
                      "sha256_actual", "hash_reason", "chain_reason", "verdict"):
            assert field in art

    def test_json_exit_code_matches_verdict(self, tmp_path, capsys):
        content = b"data"
        _write_artifact(tmp_path, "a.json", content)
        mpath = _write_manifest(tmp_path, {
            "version": "5.1", "name": "t", "description": "",
            "artifacts": [{"name": "a", "path": "a.json",
                           "sha256": _sha256(content)}],
        })
        with pytest.raises(SystemExit) as exc:
            bundle_main([str(mpath), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        expected = {PASS: 0, PARTIAL_PASS: 1, FAIL_CLOSED: 2}.get(
            data["global_verdict"], 2)
        assert exc.value.code == expected


class TestBundleMainErrors:
    def test_file_not_found_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            bundle_main(["/definitely/nonexistent/manifest.json"])
        assert exc.value.code == 2

    def test_file_not_found_prints_error_to_stderr(self, capsys):
        with pytest.raises(SystemExit):
            bundle_main(["/definitely/nonexistent/manifest.json"])
        err = capsys.readouterr().err
        assert "ERROR" in err

    def test_invalid_json_exits_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json{{{{")
        with pytest.raises(SystemExit) as exc:
            bundle_main([str(bad)])
        assert exc.value.code == 2
