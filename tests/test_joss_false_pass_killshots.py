"""Frozen JOSS false-PASS killshots.

These tests encode fail-closed contracts before implementation repair.
They intentionally preserve counterexamples discovered against the baseline.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from gpts_core.ledger import AuditLedger
from gpts_core.promotion import replay_jsonl_ledger
from omniagis.audit.bundle import BundleAuditor, FAIL_CLOSED
from omniagis.core.return_time import ReturnTimeStatistics
from omniagis.core.validator import EpsilonRobustnessValidator


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(tmp_path, data: dict):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestBundleFalsePassKillshots:
    def test_empty_manifest_fails_closed(self, tmp_path):
        manifest = _write_manifest(
            tmp_path,
            {"version": "5.1", "name": "empty", "description": "", "artifacts": []},
        )
        report = BundleAuditor().audit(str(manifest))
        assert report.global_verdict == FAIL_CLOSED

    def test_missing_artifacts_field_fails_closed(self, tmp_path):
        manifest = _write_manifest(
            tmp_path,
            {"version": "5.1", "name": "missing-artifacts", "description": ""},
        )
        report = BundleAuditor().audit(str(manifest))
        assert report.global_verdict == FAIL_CLOSED

    def test_duplicate_artifact_identity_fails_closed(self, tmp_path):
        artifact = tmp_path / "artifact.bin"
        artifact.write_bytes(b"payload")
        digest = _sha256(b"payload")
        entry = {"name": "dup", "path": "artifact.bin", "sha256": digest}
        manifest = _write_manifest(
            tmp_path,
            {
                "version": "5.1",
                "name": "duplicates",
                "description": "",
                "artifacts": [entry, dict(entry)],
            },
        )
        report = BundleAuditor().audit(str(manifest))
        assert report.global_verdict == FAIL_CLOSED

    def test_manifest_self_reference_fails_closed(self, tmp_path):
        manifest = _write_manifest(
            tmp_path,
            {
                "version": "5.1",
                "name": "self-ref",
                "description": "",
                "artifacts": [{"name": "manifest", "path": "manifest.json"}],
            },
        )
        report = BundleAuditor().audit(str(manifest))
        assert report.global_verdict == FAIL_CLOSED

    def test_stale_manifest_hash_fails_closed(self, tmp_path):
        artifact = tmp_path / "artifact.bin"
        artifact.write_bytes(b"before")
        manifest = _write_manifest(
            tmp_path,
            {
                "version": "5.1",
                "name": "stale",
                "description": "",
                "artifacts": [
                    {
                        "name": "artifact",
                        "path": "artifact.bin",
                        "sha256": _sha256(b"before"),
                    }
                ],
            },
        )
        artifact.write_bytes(b"after")
        report = BundleAuditor().audit(str(manifest))
        assert report.global_verdict == FAIL_CLOSED


class TestNumericFalsePassKillshots:
    @pytest.mark.parametrize("epsilon", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_epsilon_rejected(self, epsilon):
        with pytest.raises(ValueError):
            EpsilonRobustnessValidator(epsilon=epsilon)

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_trajectory_rejected(self, bad):
        validator = EpsilonRobustnessValidator(epsilon=0.1)
        trajectory = np.array([[0.0], [bad]])
        reference = np.array([[0.0], [0.0]])
        with pytest.raises(ValueError):
            validator.validate(trajectory, reference)

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_reference_rejected(self, bad):
        validator = EpsilonRobustnessValidator(epsilon=0.1)
        trajectory = np.array([[0.0], [0.0]])
        reference = np.array([[0.0], [bad]])
        with pytest.raises(ValueError):
            validator.validate(trajectory, reference)

    def test_empty_trajectory_rejected(self):
        validator = EpsilonRobustnessValidator(epsilon=0.1)
        with pytest.raises(ValueError):
            validator.validate(np.array([]), np.array([]))


class TestReturnTimeFalsePassKillshots:
    @pytest.mark.parametrize("tol", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_constructor_tolerance_rejected(self, tol):
        with pytest.raises(ValueError):
            ReturnTimeStatistics(tolerance=tol)

    @pytest.mark.parametrize("tol", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_override_tolerance_rejected(self, tol):
        rts = ReturnTimeStatistics(tolerance=0.1)
        with pytest.raises(ValueError):
            rts.find_returns(np.array([0.0, 1.0]), target_value=0.0, tol=tol)

    @pytest.mark.parametrize("target", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_target_rejected(self, target):
        rts = ReturnTimeStatistics(tolerance=0.1)
        with pytest.raises(ValueError):
            rts.find_returns(np.array([0.0, 1.0]), target_value=target)

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_series_rejected(self, bad):
        rts = ReturnTimeStatistics(tolerance=0.1)
        with pytest.raises(ValueError):
            rts.find_returns(np.array([0.0, bad]), target_value=0.0)

    @pytest.mark.parametrize("mean", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_stats_never_pass(self, mean):
        rts = ReturnTimeStatistics(tolerance=0.1)
        assert rts.classify({"mean": mean, "count": 10}, max_allowed_mean=10.0) == "NO PASS"

    @pytest.mark.parametrize("limit", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_classification_limit_rejected(self, limit):
        rts = ReturnTimeStatistics(tolerance=0.1)
        with pytest.raises(ValueError):
            rts.classify({"mean": 1.0, "count": 10}, max_allowed_mean=limit)


class TestLedgerFalsePassKillshots:
    def test_zero_length_chained_ledger_is_not_valid(self):
        result = AuditLedger(chained=True).verify_chain()
        assert result["valid"] is False

    def test_zero_length_jsonl_replay_fails(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("", encoding="utf-8")
        result = replay_jsonl_ledger(path)
        assert result["status"] == "FAIL"
        assert result["reason"] == "EMPTY_LEDGER"

    def test_broken_hash_chain_fails(self, tmp_path):
        e1 = {
            "event_id": "000001",
            "timestamp": "2026-09-30T00:00:00Z",
            "type": "test",
            "data": {},
            "signature": "sig",
        }
        e2 = {
            "event_id": "000002",
            "timestamp": "2026-09-30T00:00:01Z",
            "type": "test",
            "data": {"hash_parent": "0" * 64},
            "signature": "sig",
        }
        path = tmp_path / "ledger.jsonl"
        path.write_text(json.dumps(e1) + "\n" + json.dumps(e2), encoding="utf-8")
        result = replay_jsonl_ledger(path)
        assert result["status"] == "FAIL"
        assert result["reason"] == "HASH_PARENT_MISMATCH"

    def test_duplicate_event_identity_fails(self, tmp_path):
        e1 = {
            "event_id": "000001",
            "timestamp": "2026-09-30T00:00:00Z",
            "type": "test",
            "data": {},
            "signature": "sig",
        }
        e2 = dict(e1)
        path = tmp_path / "ledger.jsonl"
        path.write_text(json.dumps(e1) + "\n" + json.dumps(e2), encoding="utf-8")
        result = replay_jsonl_ledger(path)
        assert result["status"] == "FAIL"
        assert result["reason"] == "EVENT_ID_NOT_STRICTLY_INCREASING"
