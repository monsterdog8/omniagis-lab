"""Tests for MODULE_016_CORE_INIT.

Verifies that gpts_core.__init__ correctly re-exports all public symbols
and that they are callable/importable without sub-module knowledge.
"""
from __future__ import annotations

import sys
import pathlib

import pytest


class TestCoreInit:
    """Verify gpts_core top-level re-exports are present and callable."""

    # -- signals --

    def test_analyze_signal_importable(self):
        from gpts_core import analyze_signal
        assert callable(analyze_signal)

    def test_structure_score_importable(self):
        from gpts_core import structure_score
        assert callable(structure_score)

    # -- classifier --

    def test_classify_metric_importable(self):
        from gpts_core import classify_metric
        assert callable(classify_metric)

    def test_labels_is_set(self):
        from gpts_core import LABELS
        assert isinstance(LABELS, (set, frozenset, list, tuple))
        assert len(LABELS) > 0

    # -- evidence --

    def test_sha256_text_importable(self):
        from gpts_core import sha256_text
        result = sha256_text("hello")
        assert result.startswith("sha256:")

    def test_sha256_file_importable(self):
        from gpts_core import sha256_file
        assert callable(sha256_file)

    def test_build_raw_record_importable(self):
        from gpts_core import build_raw_record
        assert callable(build_raw_record)

    def test_validate_raw_records_importable(self):
        from gpts_core import validate_raw_records
        assert callable(validate_raw_records)

    # -- gate --

    def test_classify_claim_importable(self):
        from gpts_core import classify_claim
        result = classify_claim("This is a test hypothesis")
        assert hasattr(result, "status")

    def test_compute_evidence_score_importable(self):
        from gpts_core import compute_evidence_score, from_gate_counts
        inp = from_gate_counts(expected=5, present=5, valid=5)
        score = compute_evidence_score(inp)
        assert hasattr(score, "final_score")

    def test_from_gate_counts_importable(self):
        from gpts_core import from_gate_counts
        assert callable(from_gate_counts)

    def test_maturity_map_importable(self):
        from gpts_core import maturity_map
        result = maturity_map(has_raw=True, has_scoring=True, has_replay=True)
        assert isinstance(result, dict)

    def test_proof_firewall_importable(self):
        from gpts_core import proof_firewall
        assert callable(proof_firewall)

    def test_evidence_input_importable(self):
        from gpts_core import EvidenceInput
        assert EvidenceInput is not None

    def test_evidence_score_importable(self):
        from gpts_core import EvidenceScore
        assert EvidenceScore is not None

    # -- ledger --

    def test_audit_ledger_importable(self):
        from gpts_core import AuditLedger
        ledger = AuditLedger()
        ledger.log("TEST", {"x": 1})
        chain = ledger.verify_chain()
        assert chain["valid"] is True

    def test_context_citation_lock_importable(self):
        from gpts_core import ContextCitationLock
        lock = ContextCitationLock()
        assert lock.has_citation(["hello world"], "hello")

    # -- adjudication --

    def test_score_adjudication_importable(self):
        from gpts_core import score_adjudication
        rows = [{"label": "observed", "human_label": "observed"}]
        result = score_adjudication(rows)
        assert isinstance(result, dict)

    def test_build_confusion_matrix_importable(self):
        from gpts_core import build_confusion_matrix
        assert callable(build_confusion_matrix)

    # -- manifest --

    def test_build_manifest_importable(self):
        from gpts_core import build_manifest
        assert callable(build_manifest)

    def test_write_manifest_importable(self):
        from gpts_core import write_manifest
        assert callable(write_manifest)

    def test_hash_file_importable(self):
        from gpts_core import hash_file
        assert callable(hash_file)

    def test_kind_for_importable(self):
        import pathlib
        from gpts_core import kind_for
        assert kind_for(pathlib.Path("x.py")) == "python"

    # -- benchmark --

    def test_train_linear_importable(self):
        from gpts_core import train_linear
        assert callable(train_linear)

    def test_predict_linear_importable(self):
        from gpts_core import predict_linear
        assert callable(predict_linear)

    def test_mse_score_importable(self):
        from gpts_core import mse_score
        score = mse_score({"a": 0.8}, {"a": 0.8})
        assert abs(score - 1.0) < 0.01

    def test_compare_prediction_lock_importable(self):
        from gpts_core import compare_prediction_lock
        assert callable(compare_prediction_lock)

    # -- promotion --

    def test_replay_jsonl_ledger_importable(self):
        from gpts_core import replay_jsonl_ledger
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "nonexistent.jsonl"
            result = replay_jsonl_ledger(p)
            assert result["status"] == "FAIL"

    def test_evaluate_promotion_importable(self):
        from gpts_core import evaluate_promotion
        assert callable(evaluate_promotion)

    def test_validate_seal_ledger_importable(self):
        from gpts_core import validate_seal_ledger
        assert callable(validate_seal_ledger)

    # -- coherence --

    def test_shannon_entropy_importable(self):
        from gpts_core import shannon_entropy
        h = shannon_entropy([0, 1] * 50)
        assert abs(h - 1.0) < 0.01

    def test_mutual_information_importable(self):
        from gpts_core import mutual_information
        assert callable(mutual_information)

    def test_global_coherence_importable(self):
        from gpts_core import global_coherence
        c = global_coherence(0.5, 1.0)
        assert 0.0 <= c <= 1.0

    def test_build_coherence_passport_importable(self):
        from gpts_core import build_coherence_passport
        obs = {"m": [float(i) for i in range(20)]}
        passport = build_coherence_passport("C01", obs)
        assert isinstance(passport, dict)

    # -- audit_claims --

    def test_audit_claim_importable(self):
        from gpts_core import audit_claim
        assert callable(audit_claim)

    def test_audit_report_importable(self):
        from gpts_core import AuditReport
        assert AuditReport is not None

    def test_verdicts_is_list(self):
        from gpts_core import VERDICTS
        assert isinstance(VERDICTS, (list, tuple, set))

    # -- score_report --

    def test_score_audit_report_importable(self):
        from gpts_core import score_audit_report
        result = score_audit_report("")
        assert isinstance(result, dict)

    def test_evaluate_report_importable(self):
        from gpts_core import evaluate_report
        assert callable(evaluate_report)

    # -- dynamics --

    def test_continuum_state_importable(self):
        from gpts_core import ContinuumState
        s = ContinuumState()
        assert s.psi == 0.999

    def test_step_continuum_importable(self):
        from gpts_core import ContinuumState, step_continuum
        s = ContinuumState()
        s2 = step_continuum(s, noise_std=0.0)
        assert 0.0 <= s2.psi <= 1.0

    def test_continuum_energy_importable(self):
        from gpts_core import ContinuumState, continuum_energy
        s = ContinuumState()
        q = continuum_energy(s)
        assert q >= 0.0

    def test_run_continuum_importable(self):
        from gpts_core import run_continuum
        hist = run_continuum(n_steps=3, seed=0)
        assert len(hist) == 4

    def test_fractal_state_importable(self):
        from gpts_core import FractalState
        s = FractalState()
        assert s.coherence == 1.0

    def test_fractal_engine_importable(self):
        from gpts_core import FractalEngine
        engine = FractalEngine()
        state = engine.compute_metrics(cycle_id=0)
        assert 0.0 <= state.coherence <= 1.0

    # -- __all__ --

    def test_all_contains_expected_symbols(self):
        import gpts_core
        all_syms = set(gpts_core.__all__)
        required = {
            "analyze_signal", "classify_metric", "LABELS",
            "sha256_text", "classify_claim", "compute_evidence_score",
            "AuditLedger", "score_adjudication", "build_manifest",
            "train_linear", "evaluate_promotion", "shannon_entropy",
            "audit_claim", "score_audit_report",
            "ContinuumState", "FractalEngine",
        }
        missing = required - all_syms
        assert not missing, f"Missing from __all__: {missing}"
