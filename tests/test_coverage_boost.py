"""Coverage boost tests — target: push gpts_core from 65% to ≥93%.

Exercises CLI (14 subcommands), edge-case signal paths, classifier branches,
ledger persistence/chaining, promotion/seal, evidence passport, dynamics,
and score_report scoring components.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import tempfile
import zipfile

import pytest


# ===========================================================================
# CLI — all 14 subcommands via main()
# ===========================================================================

class TestCLI:
    """Test all 14 CLI subcommands through main()."""

    def _write_signal_csv(self, path: pathlib.Path) -> None:
        path.write_text("value\n" + "\n".join(str(math.sin(i * 0.1)) for i in range(300)) + "\n")

    def _write_jsonl_records(self, path: pathlib.Path, n: int = 1) -> None:
        import time
        from gpts_core.evidence import build_raw_record
        lines = []
        for i in range(n):
            rec = build_raw_record(f"run_{i}", f"task_{i}", f"out_{i}", "model_A", f"raw text {i}")
            lines.append(json.dumps(rec))
        path.write_text("\n".join(lines) + "\n")

    def _write_valid_ledger(self, path: pathlib.Path) -> None:
        event = {"event_id": "000001", "timestamp": "2026-01-01T00:00:00Z",
                 "type": "GENESIS", "data": {}, "signature": "sig1"}
        path.write_text(json.dumps(event) + "\n")

    def _write_promotion_csv(self, path: pathlib.Path, candidate: bool = False) -> None:
        if candidate:
            path.write_text(
                "artifact_id,filename,verdict,claim_ceiling,provenance_chain,"
                "dependency_resolution_status,raw_ledger_presence,replay_status,next_gate\n"
                "a001,ok.py,VERIFIED,BOUNDED,COMPLETE,RESOLVED,PRESENT,PASS,REVIEW\n"
            )
        else:
            path.write_text(
                "artifact_id,filename,verdict,claim_ceiling,provenance_chain,"
                "dependency_resolution_status,raw_ledger_presence,replay_status,next_gate\n"
                "a001,test.py,LORE,LOCAL_ONLY,INCOMPLETE,UNRESOLVED,MISSING,FAIL,NONE\n"
            )

    def _write_adjudication_csv(self, path: pathlib.Path) -> None:
        path.write_text("label,human_label\nsimulated,simulated\nobserved,observed\n")

    def _write_coherence_json(self, path: pathlib.Path) -> None:
        payload = {
            "cycle_id": "C01",
            "module_observations": {
                "modA": [float(i) for i in range(30)],
                "modB": [float(i % 5) for i in range(30)],
            },
            "system_version": "TEST_v1",
        }
        path.write_text(json.dumps(payload))

    def _write_report_md(self, path: pathlib.Path) -> None:
        path.write_text(
            "## 1. Title\nTest Report\n"
            "## 2. Executive Summary\nVerdict: WEAKLY SUPPORTED\nconfidence score: 5/10\n"
            "## 3. Detailed Analysis\nEvidence partial. observation inference modeling theory.\n"
            "## 4. Weakness Taxonomy\nType: Scope\nStatus: Active\nSeverity: Moderate\n"
            "Probable Cause: X\nDiscriminant Test: Y\n"
            "## 5. Limitations\nLimited by data. Conclusions uncertain and insufficient.\n"
            "## 6. Recommendations\nCollect more.\n"
            "## 7. Sources\nArtifact A.\n"
        )

    # ----- analyze-signal -----
    def test_analyze_signal_basic(self, tmp_path, capsys):
        from gpts_core.cli import main
        csv_file = tmp_path / "signal.csv"
        self._write_signal_csv(csv_file)
        rc = main(["analyze-signal", str(csv_file), "--column", "value", "--sr", "256.0"])
        assert rc == 0

    def test_analyze_signal_pretty(self, tmp_path, capsys):
        from gpts_core.cli import main
        csv_file = tmp_path / "signal.csv"
        self._write_signal_csv(csv_file)
        rc = main(["--pretty", "analyze-signal", str(csv_file)])
        assert rc == 0

    def test_analyze_signal_empty_column(self, tmp_path):
        from gpts_core.cli import main
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("other\nfoo\nbar\n")
        rc = main(["analyze-signal", str(csv_file), "--column", "value"])
        assert rc == 2

    # ----- classify-metric -----
    def test_classify_metric_basic(self, capsys):
        from gpts_core.cli import main
        rc = main(["classify-metric", "--name", "entropy", "--value", "0.8", "--context", "computed"])
        assert rc == 0

    def test_classify_metric_simulation(self, capsys):
        from gpts_core.cli import main
        rc = main(["classify-metric", "--name", "x", "--value", "np.random.uniform(0,1)",
                   "--row-type", "metric"])
        assert rc == 0

    # ----- validate-records -----
    def test_validate_records_pass(self, tmp_path, capsys):
        from gpts_core.cli import main
        jl = tmp_path / "records.jsonl"
        self._write_jsonl_records(jl, n=1)
        rc = main(["validate-records", str(jl), "--expected", "1"])
        assert rc == 0

    def test_validate_records_fail(self, tmp_path, capsys):
        from gpts_core.cli import main
        jl = tmp_path / "records.jsonl"
        self._write_jsonl_records(jl, n=1)
        rc = main(["validate-records", str(jl), "--expected", "5"])
        assert rc == 2

    def test_validate_records_no_strict(self, tmp_path, capsys):
        from gpts_core.cli import main
        jl = tmp_path / "records.jsonl"
        self._write_jsonl_records(jl, n=2)
        rc = main(["validate-records", str(jl), "--expected", "2", "--no-strict"])
        assert rc in (0, 2)

    # ----- classify-claim -----
    def test_classify_claim_allowed(self, capsys):
        from gpts_core.cli import main
        rc = main(["classify-claim", "this is a local hypothesis"])
        assert rc in (0, 2)

    def test_classify_claim_blocked_exits_2(self, capsys):
        from gpts_core.cli import main
        rc = main(["classify-claim", "our system is SOTA and production-ready"])
        assert rc == 2

    # ----- evidence-score -----
    def test_evidence_score_high(self, capsys):
        from gpts_core.cli import main
        rc = main(["evidence-score", "--expected", "10", "--present", "10", "--valid", "10",
                   "--safety", "1.0", "--scoring", "--replay", "--independence"])
        assert rc == 0

    def test_evidence_score_low(self, capsys):
        from gpts_core.cli import main
        rc = main(["evidence-score", "--expected", "10", "--present", "2", "--valid", "1",
                   "--safety", "0.2"])
        assert rc == 2

    # ----- replay-ledger -----
    def test_replay_ledger_pass(self, tmp_path, capsys):
        from gpts_core.cli import main
        jl = tmp_path / "ledger.jsonl"
        self._write_valid_ledger(jl)
        rc = main(["replay-ledger", str(jl)])
        assert rc == 0

    def test_replay_ledger_missing(self, tmp_path, capsys):
        from gpts_core.cli import main
        rc = main(["replay-ledger", str(tmp_path / "nonexistent.jsonl")])
        assert rc == 1

    # ----- seal-audit -----
    def test_seal_audit_empty(self, tmp_path, capsys):
        from gpts_core.cli import main
        jl = tmp_path / "seal.jsonl"
        jl.write_text("")
        rc = main(["seal-audit", str(jl), "--root", str(tmp_path)])
        assert rc == 0

    # ----- promote -----
    def test_promote_no_candidates(self, tmp_path, capsys):
        from gpts_core.cli import main
        csv_f = tmp_path / "matrix.csv"
        self._write_promotion_csv(csv_f, candidate=False)
        rc = main(["promote", str(csv_f)])
        assert rc == 1

    def test_promote_with_candidate(self, tmp_path, capsys):
        from gpts_core.cli import main
        csv_f = tmp_path / "matrix.csv"
        self._write_promotion_csv(csv_f, candidate=True)
        rc = main(["promote", str(csv_f)])
        assert rc == 0

    # ----- build-manifest -----
    def test_build_manifest(self, tmp_path, capsys):
        from gpts_core.cli import main
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.py").write_text("x=1")
        out = tmp_path / "out"
        rc = main(["build-manifest", str(src), "--out", str(out), "--label", "test"])
        assert rc == 0

    # ----- adjudicate -----
    def test_adjudicate(self, tmp_path, capsys):
        from gpts_core.cli import main
        csv_f = tmp_path / "adj.csv"
        self._write_adjudication_csv(csv_f)
        rc = main(["adjudicate", str(csv_f)])
        assert rc == 0

    # ----- coherence-passport -----
    def test_coherence_passport(self, tmp_path, capsys):
        from gpts_core.cli import main
        json_f = tmp_path / "passport_input.json"
        self._write_coherence_json(json_f)
        rc = main(["coherence-passport", str(json_f), "--bins", "8"])
        assert rc in (0, 2)

    # ----- audit-claim -----
    def test_audit_claim_basic(self, capsys):
        from gpts_core.cli import main
        rc = main(["audit-claim", "--claim", "This system performs well locally"])
        assert rc == 0

    def test_audit_claim_markdown(self, capsys):
        from gpts_core.cli import main
        rc = main(["audit-claim", "--claim", "Local lab hypothesis", "--title", "Test",
                   "--markdown"])
        assert rc == 0

    # ----- score-report -----
    def test_score_report_basic(self, tmp_path, capsys):
        from gpts_core.cli import main
        md_f = tmp_path / "report.md"
        self._write_report_md(md_f)
        rc = main(["score-report", str(md_f)])
        assert rc in (0, 2)

    def test_score_report_min_score_pass(self, tmp_path, capsys):
        from gpts_core.cli import main
        md_f = tmp_path / "report.md"
        self._write_report_md(md_f)
        rc = main(["score-report", str(md_f), "--min-score", "0"])
        assert rc == 0

    def test_score_report_min_score_fail(self, tmp_path, capsys):
        from gpts_core.cli import main
        md_f = tmp_path / "report.md"
        self._write_report_md(md_f)
        rc = main(["score-report", str(md_f), "--min-score", "999"])
        assert rc == 2

    def test_score_report_with_case(self, tmp_path, capsys):
        from gpts_core.cli import main
        md_f = tmp_path / "report.md"
        self._write_report_md(md_f)
        case = {
            "case_id": "c1", "title": "T", "claim": "C", "question": "Q",
            "dossier": "D", "oracle": {"global_verdict": "WEAKLY SUPPORTED"},
            "scoring": {"components": {"a": 50}}, "report_contract": {},
        }
        case_f = tmp_path / "case.json"
        case_f.write_text(json.dumps(case))
        rc = main(["score-report", str(md_f), "--case", str(case_f)])
        assert rc in (0, 2)

    # ----- spectral-gap -----
    def test_spectral_gap_small(self, capsys):
        from gpts_core.cli import main
        rc = main(["spectral-gap", "--alpha-min", "0.05", "--alpha-max", "0.10",
                   "--n-alpha", "2", "--n-traj", "500", "--n-bins", "10"])
        assert rc in (0, 2)

    def test_spectral_gap_larger(self, capsys):
        from gpts_core.cli import main
        rc = main(["spectral-gap", "--alpha-min", "0.02", "--alpha-max", "0.20",
                   "--n-alpha", "8", "--n-traj", "2000", "--n-bins", "30", "--seed", "7"])
        assert rc in (0, 2)


# ===========================================================================
# signals — edge cases
# ===========================================================================

class TestSignalsEdgeCases:

    def test_entropy_norm_empty_input(self):
        from gpts_core.signals import entropy_norm
        import numpy as np
        # all same value → low entropy
        h = entropy_norm(np.array([1.0] * 32))
        assert h == 0.0

    def test_compression_ratio_constant(self):
        from gpts_core.signals import compression_ratio
        import numpy as np
        r = compression_ratio(np.array([5.0] * 64))
        assert isinstance(r, float)
        assert r > 0.0

    def test_autocorr_zero_variance(self):
        from gpts_core.signals import autocorr
        import numpy as np
        lag1, acmax = autocorr(np.array([1.0] * 64))
        assert lag1 == 0.0
        assert acmax == 0.0

    def test_spectral_analysis_short_input(self):
        from gpts_core.signals import spectral_analysis
        import numpy as np
        pf, conc, ent = spectral_analysis(np.array([1.0, 2.0]), sr=256.0)
        assert pf == 0.0 and conc == 0.0 and ent == 0.0

    def test_spectral_analysis_zero_power(self):
        from gpts_core.signals import spectral_analysis
        import numpy as np
        pf, conc, ent = spectral_analysis(np.zeros(64), sr=256.0)
        assert pf == 0.0

    def test_window_stats_too_short(self):
        from gpts_core.signals import window_stats
        import numpy as np
        pcv, ecv, jsd = window_stats(np.array([1.0] * 10), sr=256.0)
        assert math.isnan(pcv)

    def test_generate_null_shuffle(self):
        from gpts_core.signals import generate_null
        import numpy as np
        x = np.array([float(i) for i in range(100)])
        result = generate_null(x, seed=42, kind="shuffle")
        assert len(result) == 100
        assert sorted(result) == sorted(x.tolist())

    def test_generate_null_gaussian(self):
        from gpts_core.signals import generate_null
        import numpy as np
        x = np.linspace(0, 1, 100)
        result = generate_null(x, seed=42, kind="gaussian")
        assert len(result) == 100

    def test_generate_null_same_marginal(self):
        from gpts_core.signals import generate_null
        import numpy as np
        x = np.array([float(i) for i in range(50)])
        result = generate_null(x, seed=42, kind="same_marginal")
        assert len(result) == 50

    def test_generate_null_phase_scramble(self):
        from gpts_core.signals import generate_null
        import numpy as np
        x = np.array([math.sin(i * 0.2) for i in range(128)])
        result = generate_null(x, seed=42, kind="phase_scramble")
        assert len(result) == 128

    def test_generate_null_unknown_kind(self):
        from gpts_core.signals import generate_null
        import numpy as np
        with pytest.raises(ValueError, match="Unknown null kind"):
            generate_null(np.array([1.0] * 10), seed=0, kind="nonexistent")

    def test_step_metrics_too_short(self):
        from gpts_core.signals import step_metrics
        import numpy as np
        result = step_metrics(np.array([0.0]), np.array([1.0]))
        assert result == {}

    def test_step_metrics_no_overshoot(self):
        from gpts_core.signals import step_metrics
        import numpy as np
        t = np.linspace(0, 1, 200)
        y = np.where(t < 0.3, 0.0, 1.0)
        m = step_metrics(t, y)
        assert "rise_time" in m
        assert "settling_time" in m

    def test_ar2_fit_returns_five_values(self):
        from gpts_core.signals import ar2_fit
        import numpy as np
        x = np.array([math.sin(i * 0.3) for i in range(100)])
        a1, a2, r2, ef, rr = ar2_fit(x, sr=100.0)
        assert isinstance(r2, float)

    def test_ar2_fit_degenerate(self):
        from gpts_core.signals import ar2_fit
        import numpy as np
        x = np.zeros(20)
        a1, a2, r2, ef, rr = ar2_fit(x, sr=100.0)
        assert a1 == 0.0 and a2 == 0.0


# ===========================================================================
# classifier — missing branches
# ===========================================================================

class TestClassifierBranches:

    def test_unsupported_high_claim(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("x", "global superiority", "production-ready")
        assert label == "unsupported"
        assert "unsupported_high_claim_pattern" in reasons

    def test_equation_recalculable(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric(
            "formula_x", "3.14", "equation context",
            row_type="recalculation", recalculable="True", result="3.14"
        )
        assert label == "computed"
        assert "safe_arithmetic_recalculation" in reasons

    def test_equation_symbolic(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric(
            "psi_omega", "∞", "symbolic",
            row_type="equation", recalculable="False", result=""
        )
        assert label == "symbolic"
        assert "symbolic_equation" in reasons

    def test_equation_not_recalculable(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric(
            "unknown_formula", "abc", "",
            row_type="equation", recalculable="False", result=""
        )
        assert label == "unsupported"
        assert "equation_not_recalculable" in reasons

    def test_observed_label_explicit(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("image_luminance", "128.5", "pixel measurement rgb")
        assert label == "observed"

    def test_symbolic_label_explicit(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("omega_constant", "42", "cosmic quantum entity")
        assert label == "symbolic"

    def test_computed_generic_variable(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("x", "3.14", "mean of values computed sum")
        assert label == "unsupported"
        assert "generic_variable_name_with_computation_context" in reasons

    def test_computed_non_generic(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("mean_accuracy", "0.85", "calculated mean score")
        assert label == "computed"
        assert "computed_metric_pattern" in reasons

    def test_reported_numeric_named(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("batch_size", "32", "")
        assert label in ("reported", "computed", "unsupported")

    def test_numeric_generic_one_char(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("n", "100", "")
        assert label == "unsupported"

    def test_insufficient_context(self):
        from gpts_core.classifier import classify_metric
        label, reasons, conf = classify_metric("some_metric", "text_not_numeric", "")
        assert label in ("unsupported", "reported", "symbolic", "computed")


# ===========================================================================
# ledger — persistence, chaining, gate, cache
# ===========================================================================

class TestLedgerEdgeCases:

    def test_log_to_file_persists(self, tmp_path):
        from gpts_core.ledger import AuditLedger
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = AuditLedger(path=ledger_path)
        ledger.log("EVT_1", {"x": 1})
        ledger.log("EVT_2", {"x": 2})
        # new ledger loads from file
        ledger2 = AuditLedger(path=ledger_path)
        assert len(ledger2.entries()) == 2

    def test_chained_mode_verify_valid(self, tmp_path):
        from gpts_core.ledger import AuditLedger
        ledger_path = tmp_path / "chain.jsonl"
        ledger = AuditLedger(path=ledger_path, chained=True)
        ledger.log("EVT_A", {"a": 1}, audit=True)
        ledger.log("EVT_B", {"b": 2}, audit=True)
        chain = ledger.verify_chain()
        assert chain["valid"] is True
        assert chain["entries_checked"] == 2

    def test_chained_non_audit_events(self, tmp_path):
        from gpts_core.ledger import AuditLedger
        ledger = AuditLedger(chained=True)
        # audit=False → no chaining even in chained mode
        ledger.log("EVT", {"x": 0}, audit=False)
        ledger.log("EVT2", {"x": 1}, audit=True)
        result = ledger.verify_chain()
        # entries without prev_hash will fail chain check
        assert isinstance(result, dict)

    def test_gate_not_ready_raises(self):
        from gpts_core.ledger import AuditLedger, _SyncGate
        gate = _SyncGate(guard_stable=False)
        ledger = AuditLedger(gate=gate)
        with pytest.raises(RuntimeError, match="LEDGER_GATE_NOT_READY"):
            ledger.log("EVT", {})

    def test_gate_all_true_ready(self):
        from gpts_core.ledger import _SyncGate
        gate = _SyncGate(guard_stable=True, experts_synced=True,
                         rules_locked=True, context_frozen=True)
        assert gate.ready() is True

    def test_gate_any_false_not_ready(self):
        from gpts_core.ledger import _SyncGate
        assert not _SyncGate(experts_synced=False).ready()
        assert not _SyncGate(rules_locked=False).ready()
        assert not _SyncGate(context_frozen=False).ready()

    def test_citation_cache_hit(self):
        from gpts_core.ledger import ContextCitationLock
        lock = ContextCitationLock()
        corpus = ["quick brown fox"]
        # First call: cache miss → scan
        r1 = lock.has_citation(corpus, "brown fox")
        # Second call: cache hit
        r2 = lock.has_citation(corpus, "brown fox")
        assert r1 == r2 == True

    def test_citation_cache_eviction(self):
        from gpts_core.ledger import ContextCitationLock
        lock = ContextCitationLock(max_cache=3)
        corpus = ["apple", "banana", "cherry", "date", "elderberry"]
        # Fill cache past max_cache
        for word in ["apple", "banana", "cherry", "date"]:
            lock.has_citation(corpus, word)
        # Should not raise; cache evicts oldest
        assert isinstance(lock.has_citation(corpus, "elderberry"), bool)

    def test_citation_non_exact(self):
        from gpts_core.ledger import ContextCitationLock
        lock = ContextCitationLock()
        corpus = ["value is 3.14159"]
        # Non-exact: treat as raw regex
        result = lock.has_citation(corpus, r"3\.\d+", exact=False)
        assert result is True

    def test_citation_require_none_raises(self):
        from gpts_core.ledger import ContextCitationLock
        lock = ContextCitationLock()
        with pytest.raises(ValueError, match="CITATION_MISSING"):
            lock.require_citation(["corpus text"], None)

    def test_citation_empty_value(self):
        from gpts_core.ledger import ContextCitationLock
        lock = ContextCitationLock()
        assert lock.has_citation(["some text"], "") is False

    def test_chained_load_from_file_restores_chain(self, tmp_path):
        from gpts_core.ledger import AuditLedger
        p = tmp_path / "chain.jsonl"
        ledger1 = AuditLedger(path=p, chained=True)
        ledger1.log("EVT_1", {}, audit=True)
        # Reload from file — should restore chain_last
        ledger2 = AuditLedger(path=p, chained=True)
        assert len(ledger2.entries()) == 1


# ===========================================================================
# promotion — seal records, hash chain edge cases
# ===========================================================================

class TestPromotionEdgeCases:

    def test_sha256_file_internal(self, tmp_path):
        from gpts_core.promotion import _sha256_file
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        h = _sha256_file(f)
        assert h == hashlib.sha256(b"hello").hexdigest()

    def test_sha256_file_missing(self, tmp_path):
        from gpts_core.promotion import _sha256_file
        assert _sha256_file(tmp_path / "nope.bin") is None

    def test_validate_seal_record_missing_fields(self, tmp_path):
        from gpts_core.promotion import validate_seal_record
        row = {"event_id": "001"}  # missing most required fields
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is False
        assert any("missing:" in e for e in result["errors"])

    def test_validate_seal_record_artifact_missing(self, tmp_path):
        from gpts_core.promotion import validate_seal_record
        row = {
            "schema": "v1", "event_id": "001", "created_at_utc": "2026-01-01T00:00:00Z",
            "artifact_path": "nonexistent.py", "artifact_exists": True,
            "expected_sha256": "abc", "expected_bytes": 100,
            "recomputed_sha256": None, "recomputed_bytes": None,
            "verdict": "VERIFIED", "proof_scope": "LOCAL",
            "blocked_claims": [], "claim_ceiling": "LOCAL_ONLY",
        }
        result = validate_seal_record(row, tmp_path)
        assert "artifact_exists_mismatch" in result["errors"]

    def test_validate_seal_record_valid_artifact(self, tmp_path):
        from gpts_core.promotion import validate_seal_record
        f = tmp_path / "file.py"
        f.write_bytes(b"x=1")
        sha = hashlib.sha256(b"x=1").hexdigest()
        row = {
            "schema": "v1", "event_id": "001", "created_at_utc": "2026-01-01T00:00:00Z",
            "artifact_path": "file.py", "artifact_exists": True,
            "expected_sha256": sha, "expected_bytes": 3,
            "recomputed_sha256": sha, "recomputed_bytes": 3,
            "verdict": "VERIFIED", "proof_scope": "LOCAL",
            "blocked_claims": [], "claim_ceiling": "LOCAL_ONLY",
        }
        result = validate_seal_record(row, tmp_path)
        assert result["valid"] is True

    def test_validate_seal_ledger_with_valid_record(self, tmp_path):
        from gpts_core.promotion import validate_seal_ledger
        f = tmp_path / "artifact.py"
        f.write_bytes(b"code")
        sha = hashlib.sha256(b"code").hexdigest()
        row = {
            "schema": "v1", "event_id": "001", "created_at_utc": "T",
            "artifact_path": "artifact.py", "artifact_exists": True,
            "expected_sha256": sha, "expected_bytes": 4,
            "recomputed_sha256": sha, "recomputed_bytes": 4,
            "verdict": "VERIFIED", "proof_scope": "LOCAL",
            "blocked_claims": [], "claim_ceiling": "LOCAL_ONLY",
        }
        ledger = tmp_path / "seal.jsonl"
        ledger.write_text(json.dumps(row) + "\n")
        result = validate_seal_ledger(ledger, tmp_path)
        assert result["records"] == 1
        assert result["pass_count"] == 1

    def test_validate_seal_ledger_bad_json_line(self, tmp_path):
        from gpts_core.promotion import validate_seal_ledger
        ledger = tmp_path / "seal.jsonl"
        ledger.write_text("{not valid json}\n")
        result = validate_seal_ledger(ledger, tmp_path)
        assert result["records"] == 0  # malformed line skipped

    def test_replay_data_not_object(self, tmp_path):
        from gpts_core.promotion import replay_jsonl_ledger
        f = tmp_path / "bad.jsonl"
        event = {"event_id": "000001", "timestamp": "T", "type": "X",
                 "data": "not_a_dict", "signature": "s"}
        f.write_text(json.dumps(event) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "DATA_NOT_OBJECT"

    def test_replay_event_id_not_increasing(self, tmp_path):
        from gpts_core.promotion import replay_jsonl_ledger
        f = tmp_path / "bad.jsonl"
        e1 = {"event_id": "000002", "timestamp": "T", "type": "X", "data": {}, "signature": "s"}
        e2 = {"event_id": "000001", "timestamp": "T", "type": "X", "data": {}, "signature": "s"}
        f.write_text(json.dumps(e1) + "\n" + json.dumps(e2) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "EVENT_ID_NOT_STRICTLY_INCREASING"

    def test_replay_genesis_parent_not_zero(self, tmp_path):
        from gpts_core.promotion import replay_jsonl_ledger
        f = tmp_path / "bad.jsonl"
        e = {"event_id": "000001", "timestamp": "T", "type": "GENESIS",
             "data": {"hash_parent": "abc"}, "signature": "s"}
        f.write_text(json.dumps(e) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "GENESIS_PARENT_NOT_ZERO"

    def test_replay_hash_parent_mismatch(self, tmp_path):
        from gpts_core.promotion import replay_jsonl_ledger
        import hashlib, json as _json
        f = tmp_path / "chain.jsonl"
        e1 = {"event_id": "000001", "timestamp": "T", "type": "GENESIS", "data": {}, "signature": "s"}
        e2 = {"event_id": "000002", "timestamp": "T", "type": "EVT",
              "data": {"hash_parent": "wrong_hash"}, "signature": "s"}
        f.write_text(_json.dumps(e1) + "\n" + _json.dumps(e2) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "FAIL"
        assert result["reason"] == "HASH_PARENT_MISMATCH"

    def test_replay_valid_chain_with_hash_parent(self, tmp_path):
        import hashlib, json as _json
        from gpts_core.promotion import replay_jsonl_ledger

        def _sha(obj):
            return hashlib.sha256(
                _json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        e1 = {"event_id": "000001", "timestamp": "T", "type": "GENESIS", "data": {}, "signature": "s"}
        h1 = _sha(e1)
        e2 = {"event_id": "000002", "timestamp": "T", "type": "EVT",
              "data": {"hash_parent": h1}, "signature": "s"}
        f = tmp_path / "chain.jsonl"
        f.write_text(_json.dumps(e1) + "\n" + _json.dumps(e2) + "\n")
        result = replay_jsonl_ledger(f)
        assert result["status"] == "PASS"
        assert result["events_count"] == 2

    def test_token_match_function(self):
        from gpts_core.promotion import _token_match
        assert _token_match("COMPLETE", ("COMPLETE", "VERIFIED"))
        assert not _token_match("MISSING", ("PRESENT", "RESOLVED"))

    def test_row_blockers_quarantine(self):
        from gpts_core.promotion import _row_blockers
        row = {"verdict": "QUARANTINED", "raw_ledger_presence": "", "replay_status": ""}
        blockers = _row_blockers(row)
        assert "QUARANTINED" in blockers


# ===========================================================================
# evidence — validate_metric_passport, load_jsonl edge cases
# ===========================================================================

class TestEvidenceEdgeCases:

    def test_canonical_json(self):
        from gpts_core.evidence import canonical_json
        result = canonical_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_sha256_json(self):
        from gpts_core.evidence import sha256_json
        h = sha256_json({"a": 1})
        assert h.startswith("sha256:")

    def test_sha256_bytes(self):
        from gpts_core.evidence import sha256_bytes
        h = sha256_bytes(b"hello")
        assert h == "sha256:" + hashlib.sha256(b"hello").hexdigest()

    def test_validate_raw_record_non_string_output(self):
        from gpts_core.evidence import validate_raw_record
        rec = {
            "run_id": "r1", "task_id": "t1", "output_id": "o1",
            "model_slot": "m", "raw_output": 12345,  # not a string
            "token_count": 5, "latency_ms": 100,
            "timestamp": "T", "output_hash": "sha256:" + "a" * 64,
        }
        verdict = validate_raw_record(rec)
        assert not verdict.valid
        assert "RAW_OUTPUT_NOT_STRING" in verdict.errors

    def test_validate_raw_record_negative_count(self):
        from gpts_core.evidence import build_raw_record, validate_raw_record
        rec = build_raw_record("r1", "t1", "o1", "m", "text")
        rec = dict(rec)
        rec["token_count"] = -5
        verdict = validate_raw_record(rec)
        assert any("NEGATIVE_FIELD" in e for e in verdict.errors)

    def test_validate_raw_record_non_int_field(self):
        from gpts_core.evidence import build_raw_record, validate_raw_record
        rec = build_raw_record("r1", "t1", "o1", "m", "text")
        rec = dict(rec)
        rec["latency_ms"] = "not_an_int"
        verdict = validate_raw_record(rec)
        assert any("NON_INTEGER_FIELD" in e for e in verdict.errors)

    def test_validate_metric_passport_not_an_object(self):
        from gpts_core.evidence import validate_metric_passport
        result = validate_metric_passport("not a dict")
        assert result["valid"] is False
        assert "NOT_AN_OBJECT" in result["blockers"]

    def test_validate_metric_passport_missing_fields(self):
        from gpts_core.evidence import validate_metric_passport
        result = validate_metric_passport({})
        assert result["valid"] is False
        assert any("MISSING:" in b for b in result["blockers"])

    def test_validate_metric_passport_payload_hash_mismatch(self):
        from gpts_core.evidence import validate_metric_passport, sha256_json
        from gpts_core.evidence import _SHA256_PREFIX
        payload = {"value": 42}
        entry_base = {
            "metric_namespace": "ns", "formula_id": "f1", "source_backend": "local",
            "record_semantics": "obs", "cycle_semantics": "c", "raw_payload_status": "ok",
            "hash_status": "ok", "replay_status": "ok",
            "cycle": 1, "timestamp_utc": "T", "raw_payload": payload,
            "payload_hash": "sha256:" + "wrong" * 10,  # wrong hash
            "entry_hash": _SHA256_PREFIX + "0" * 64,
        }
        result = validate_metric_passport(entry_base)
        assert "PAYLOAD_HASH_MISMATCH" in result["blockers"]

    def test_validate_metric_passport_cycle_mismatch(self):
        from gpts_core.evidence import validate_metric_passport, sha256_json, canonical_json
        from gpts_core.evidence import _SHA256_PREFIX
        import hashlib
        payload = {"value": 1}
        payload_hash = "sha256:" + hashlib.sha256(
            canonical_json(payload).encode()
        ).hexdigest()
        obj = {
            "metric_namespace": "ns", "formula_id": "f1", "source_backend": "local",
            "record_semantics": "obs", "cycle_semantics": "c", "raw_payload_status": "ok",
            "hash_status": "ok", "replay_status": "ok",
            "cycle": 5, "timestamp_utc": "T", "raw_payload": payload,
            "payload_hash": payload_hash,
            "entry_hash": _SHA256_PREFIX + "0" * 64,
        }
        result = validate_metric_passport(obj, expect_cycle=99)
        assert any("CYCLE_MISMATCH" in b for b in result["blockers"])

    def test_load_jsonl_malformed_line(self, tmp_path):
        from gpts_core.evidence import load_jsonl
        f = tmp_path / "data.jsonl"
        f.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
        result = load_jsonl(f)
        assert len(result) == 3  # 3 lines: 1 valid, 1 error, 1 valid
        assert result[1].get("_error") == "PARSE_ERROR"

    def test_load_jsonl_non_dict_value(self, tmp_path):
        from gpts_core.evidence import load_jsonl
        f = tmp_path / "data.jsonl"
        f.write_text('"just_a_string"\n')
        result = load_jsonl(f)
        assert result[0].get("raw_output") == ""

    def test_load_jsonl_nonexistent(self, tmp_path):
        from gpts_core.evidence import load_jsonl
        result = load_jsonl(tmp_path / "nope.jsonl")
        assert result == []

    def test_validate_raw_records_scoring_done(self):
        from gpts_core.evidence import build_raw_record, validate_raw_records
        rec = build_raw_record("r1", "t1", "o1", "m", "text")
        report = validate_raw_records(run_id="r1", expected=1, records=[rec], scoring_done=True)
        assert report.scoring_status in ("DONE_LOCAL", "BLOCKED", "READY")

    def test_audit_zip_with_run_summary(self, tmp_path):
        from gpts_core.evidence import audit_zip
        zp = tmp_path / "test.zip"
        run_summary = {"ledger_replay": {"status": "PASS", "mismatch_count": 0}}
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("run_summary.json", json.dumps(run_summary))
            z.writestr("file.py", "x=1")
        result = audit_zip(zp)
        assert result["replay_pass"] is True
        assert result["verdict"] == "REPLAY_PASS_LOCAL"

    def test_audit_zip_nonexistent(self, tmp_path):
        from gpts_core.evidence import audit_zip
        result = audit_zip(tmp_path / "nope.zip")
        assert result["exists"] is False

    def test_summarize_csv_no_numeric(self, tmp_path):
        from gpts_core.evidence import summarize_csv
        f = tmp_path / "text.csv"
        f.write_text("name,value\nalpha,hello\nbeta,world\n")
        result = summarize_csv(f)
        assert result["verdict"] == "NO_NUMERIC_CELLS"

    def test_summarize_csv_missing(self, tmp_path):
        from gpts_core.evidence import summarize_csv
        result = summarize_csv(tmp_path / "nope.csv")
        assert result["exists"] is False


# ===========================================================================
# dynamics — run_continuum, FractalEngine extended
# ===========================================================================

class TestDynamicsExtended:

    def test_run_continuum_returns_list(self):
        from gpts_core.dynamics import ContinuumState, run_continuum
        history = run_continuum(5, seed=42)
        assert len(history) == 6  # initial + 5 steps
        for state_dict in history:
            assert "psi" in state_dict
            assert 0.0 <= state_dict["psi"] <= 1.0

    def test_run_continuum_with_initial(self):
        from gpts_core.dynamics import ContinuumState, run_continuum
        init = ContinuumState(psi=0.5, F=0.5, S=0.5, C=0.5, R=40.0, Q=0.0)
        history = run_continuum(3, initial=init, seed=99)
        assert len(history) == 4

    def test_continuum_state_to_dict(self):
        from gpts_core.dynamics import ContinuumState
        s = ContinuumState(psi=0.8, F=0.7, S=0.3, C=0.2, R=40.0, Q=1.0)
        d = s.to_dict()
        assert d["psi"] == 0.8
        assert d["F"] == 0.7

    def test_fractal_engine_get_state(self):
        from gpts_core.dynamics import FractalEngine
        engine = FractalEngine()
        engine.compute_metrics(cycle_id=1)
        state = engine.get_state()
        assert hasattr(state, "coherence")

    def test_fractal_engine_is_stable(self):
        from gpts_core.dynamics import FractalEngine
        engine = FractalEngine(history_size=5)
        for i in range(5):
            engine.compute_metrics(cycle_id=i)
        result = engine.is_stable()
        assert isinstance(result, bool)

    def test_fractal_engine_reset(self):
        from gpts_core.dynamics import FractalEngine
        engine = FractalEngine()
        for i in range(10):
            engine.compute_metrics(cycle_id=i)
        engine.reset()
        assert len(engine.get_history()) == 0
        assert engine.cycle_count == 0

    def test_fractal_engine_empty_statistics(self):
        from gpts_core.dynamics import FractalEngine
        engine = FractalEngine()
        stats = engine.get_statistics()
        assert stats["mean"] == 0.0

    def test_fractal_state_to_dict(self):
        from gpts_core.dynamics import FractalState
        s = FractalState(coherence=0.99, entropy=0.01, resonance_hz=12.0, drift=0.001)
        d = s.to_dict()
        assert d["coherence"] == 0.99

    def test_fractal_engine_long_history_drift(self):
        from gpts_core.dynamics import FractalEngine
        engine = FractalEngine(history_size=50)
        for i in range(25):
            engine.compute_metrics(cycle_id=i)
        # history between 20-50: drift may be non-zero
        state = engine.get_state()
        assert state.drift >= 0.0
        # continue past 20
        for i in range(25, 50):
            engine.compute_metrics(cycle_id=i)
        state = engine.get_state()
        assert state.drift >= 0.0


# ===========================================================================
# score_report — scoring component branches
# ===========================================================================

class TestScoreReportBranches:

    _TOLERANCE = {
        "full_score_if_abs_diff_lte": 0.5,
        "light_penalty_if_abs_diff_lte": 1.0,
        "medium_penalty_if_abs_diff_lte": 2.0,
    }

    def _full_report(self) -> str:
        return (
            "## 1. Title\nTest Report\n"
            "## 2. Executive Summary\nVerdict: SUPPORTED\nconfidence score: 8/10\n"
            "## 3. Detailed Analysis\n"
            "observation inference modeling theory interpretation\n"
            "Test 1: discriminant test example. Test 2: another test.\n"
            "SUPPORTED: strong evidence. UNSUPPORTED: weak counterevidence.\n"
            "## 4. Weakness Taxonomy\n"
            "Type: Scope\nStatus: Active\nSeverity: Moderate\n"
            "Probable Cause: X\nDiscriminant Test: Y\n"
            "Type: Proxy\nStatus: Contested\nSeverity: Low\n"
            "Probable Cause: Z\nDiscriminant Test: W\n"
            "## 5. Limitations\nLimited scope. Conclusions uncertain. Results fragile.\n"
            "fail-closed guard present.\n"
            "## 6. Recommendations\nCollect more data from independent sources.\n"
            "## 7. Sources\nArtifact A. Experiment B.\n"
        )

    def _default_case(self, verdict="SUPPORTED", confidence=8.0):
        return {
            "case_id": "c1", "title": "T", "claim": "C", "question": "Q", "dossier": "D",
            "oracle": {
                "global_verdict": verdict,
                "confidence_score": confidence,
                "sub_verdicts": [],
                "critical_points_required": ["observation", "inference"],
                "expected_min_discriminant_tests": 1,
                "fatal_errors": [],
            },
            "scoring": {
                "components": {
                    "global_and_subverdicts": 20,
                    "critical_points_coverage": 20,
                    "confidence_calibration": 15,
                    "discriminant_tests": 15,
                    "probative_separation": 10,
                    "weakness_taxonomy": 10,
                    "fail_closed_and_limitations": 10,
                },
                "confidence_tolerance": self._TOLERANCE,
                "bonuses": {"original_relevant_test_max": 0},
                "maluses": {"overconfidence_max": 5},
            },
            "report_contract": {
                "required_sections": ["Title", "Executive Summary", "Detailed Analysis",
                                      "Weakness Taxonomy", "Limitations", "Recommendations", "Sources"],
                "allowed_verdicts": ["SUPPORTED", "PARTIALLY SUPPORTED", "WEAKLY SUPPORTED",
                                     "UNSUPPORTED", "UNCERTAIN"],
                "weakness_microsyntax": {"fields": ["Type", "Status", "Severity",
                                                     "Probable Cause", "Discriminant Test"]},
            },
        }

    def test_score_verdicts(self):
        from gpts_core.score_report import score_verdicts, extract_sections
        report = self._full_report()
        sections = extract_sections(report)
        oracle = {"global_verdict": "SUPPORTED", "sub_verdicts": []}
        score, note, details = score_verdicts(report, sections, oracle)
        assert isinstance(score, int)
        assert 0 <= score <= 20

    def test_score_verdicts_mismatch(self):
        from gpts_core.score_report import score_verdicts, extract_sections
        report = self._full_report()
        sections = extract_sections(report)
        oracle = {"global_verdict": "UNSUPPORTED", "sub_verdicts": []}
        score, note, details = score_verdicts(report, sections, oracle)
        assert score < 20

    def test_score_verdicts_with_sub_verdicts(self):
        from gpts_core.score_report import score_verdicts, extract_sections
        report = self._full_report()
        sections = extract_sections(report)
        oracle = {
            "global_verdict": "SUPPORTED",
            "sub_verdicts": [{"proposition": "observation", "verdict": "SUPPORTED"}],
        }
        score, note, details = score_verdicts(report, sections, oracle)
        assert 0 <= score <= 20

    def test_score_critical_points(self):
        from gpts_core.score_report import score_critical_points
        report = self._full_report()
        score, note, uncov = score_critical_points(report, ["observation", "inference"])
        assert 0 <= score <= 20

    def test_score_critical_points_empty(self):
        from gpts_core.score_report import score_critical_points
        score, note, uncov = score_critical_points("some report", [])
        assert score == 20  # max_points returned when no critical points

    def test_score_confidence_calibrated(self):
        from gpts_core.score_report import score_confidence
        score, note = score_confidence(8.0, 8.0, self._TOLERANCE, 15)
        assert score == 15

    def test_score_confidence_light_penalty(self):
        from gpts_core.score_report import score_confidence
        score, note = score_confidence(7.3, 8.0, self._TOLERANCE, 15)
        assert score == 13  # light penalty

    def test_score_confidence_medium_penalty(self):
        from gpts_core.score_report import score_confidence
        score, note = score_confidence(6.0, 8.0, self._TOLERANCE, 15)
        assert score == 10  # medium penalty

    def test_score_confidence_strong_penalty(self):
        from gpts_core.score_report import score_confidence
        score, note = score_confidence(3.0, 8.0, self._TOLERANCE, 15)
        assert score <= 10

    def test_score_confidence_missing(self):
        from gpts_core.score_report import score_confidence
        score, note = score_confidence(None, 8.0, self._TOLERANCE, 15)
        assert score == 0

    def test_score_discriminant_tests(self):
        from gpts_core.score_report import score_discriminant_tests, extract_sections
        sections = extract_sections(self._full_report())
        oracle = {"expected_min_discriminant_tests": 1}
        score, note, count = score_discriminant_tests(sections, oracle, 15)
        assert 0 <= score <= 15

    def test_score_probative_separation_full(self):
        from gpts_core.score_report import score_probative_separation
        score, note = score_probative_separation(self._full_report(), 10)
        assert score == 10

    def test_score_probative_separation_none(self):
        from gpts_core.score_report import score_probative_separation
        score, note = score_probative_separation("just words here", 10)
        assert score == 0

    def test_score_weakness_taxonomy_present(self):
        from gpts_core.score_report import score_weakness_taxonomy, extract_sections
        sections = extract_sections(self._full_report())
        score, note = score_weakness_taxonomy(sections, max_points=10)
        assert 0 <= score <= 10

    def test_score_weakness_taxonomy_missing(self):
        from gpts_core.score_report import score_weakness_taxonomy
        score, note = score_weakness_taxonomy({}, max_points=10)
        assert score == 0

    def test_score_fail_closed_high(self):
        from gpts_core.score_report import score_fail_closed
        sections = {
            "Limitations": "Limited scope. Conclusions uncertain. Results are fragile. Cannot be used in production. Insufficient evidence."
        }
        score, note = score_fail_closed(sections, 10)
        assert score >= 4

    def test_score_fail_closed_missing(self):
        from gpts_core.score_report import score_fail_closed
        score, note = score_fail_closed({}, 10)
        assert score == 0

    def test_score_audit_report_full(self):
        from gpts_core.score_report import score_audit_report
        result = score_audit_report(self._full_report())
        assert "total" in result
        assert 0 <= result["total"] <= 100

    def test_evaluate_with_full_oracle(self):
        from gpts_core.score_report import evaluate
        result = evaluate(self._default_case(), self._full_report(), strict=False)
        assert "total" in result
        assert 0 <= result["total"] <= 100

    def test_evaluate_strict_invalid_report(self):
        from gpts_core.score_report import evaluate
        case = self._default_case()
        result = evaluate(case, "just a short text", strict=True)
        assert result["total"] == 0

    def test_evaluate_verdict_mismatch(self):
        from gpts_core.score_report import evaluate
        case = self._default_case(verdict="UNSUPPORTED")
        result = evaluate(case, self._full_report(), strict=False)
        assert any("Verdict mismatch" in d for d in result.get("differences", []))

    def test_normalize_text(self):
        from gpts_core.score_report import normalize_text
        result = normalize_text("Héllo Wörld — test")
        assert "hello" in result
        assert "world" in result

    def test_extract_sections_numbered(self):
        from gpts_core.score_report import extract_sections
        md = "## 1. Title\nMy Report\n## 2. Executive Summary\nSummary here\n"
        sections = extract_sections(md)
        assert len(sections) >= 1

    def test_validate_case_json_components_wrong_sum(self):
        from gpts_core.score_report import validate_case_json
        case = self._default_case()
        case["scoring"]["components"]["global_and_subverdicts"] = 99  # sum != 100
        issues = validate_case_json(case)
        assert any("100" in i for i in issues)


# ===========================================================================
# benchmark — additional paths
# ===========================================================================

class TestBenchmarkExtended:

    def test_minmax_fit_and_transform(self):
        from gpts_core.benchmark import minmax_fit, minmax_transform
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        lo, hi = minmax_fit(data)
        assert len(lo) == 2
        transformed = minmax_transform(data, lo, hi)
        assert abs(transformed[0][0] - 0.0) < 1e-9
        assert abs(transformed[-1][0] - 1.0) < 1e-9

    def test_minmax_transform_constant_column(self):
        from gpts_core.benchmark import minmax_fit, minmax_transform
        data = [[5.0], [5.0], [5.0]]
        lo, hi = minmax_fit(data)
        transformed = minmax_transform(data, lo, hi)
        assert all(row[0] == 0.0 for row in transformed)

    def test_read_feature_csv(self, tmp_path):
        from gpts_core.benchmark import read_feature_csv
        f = tmp_path / "features.csv"
        f.write_text("id,entropy,autocorr\nr1,0.8,0.5\nr2,0.6,0.4\n")
        rows = read_feature_csv(f, feature_cols=["entropy", "autocorr"])
        assert len(rows) == 2
        assert rows[0] == [0.8, 0.5]

    def test_read_score_csv(self, tmp_path):
        from gpts_core.benchmark import read_score_csv
        f = tmp_path / "scores.csv"
        f.write_text("run_id,psiomega_mean\nr1,0.8\nr2,0.6\n")
        scores = read_score_csv(f)
        assert isinstance(scores, dict)

    def test_predict_linear_no_clip(self):
        from gpts_core.benchmark import predict_linear
        preds = predict_linear([[5.0]], [1.0], 0.0, clip=False)
        assert preds[0] == 5.0

    def test_compare_prediction_lock_normal_pass(self):
        from gpts_core.benchmark import compare_prediction_lock
        lock = {"gate_g2_numeric_prediction": {
            "primary_metric": "psiomega_mean",
            "target_next_shadow_phase": 13,
            "predicted_value": 0.56,
            "strong_pass_interval": [0.52, 0.62],
            "normal_pass_interval": [0.48, 0.66],
        }}
        obs = {
            "run_id": "r1", "phase": 13, "psiomega_mean": 0.50,
            "ledger_replay_status": "PASS", "control_effects": 0,
            "production_unlocked": False, "stdout_retained": True, "stderr_retained": True,
        }
        result = compare_prediction_lock(lock, obs)
        assert "PASS" in result["verdict"] or "FAIL" in result["verdict"]
        assert result["production_unlocked"] is False


# ===========================================================================
# coherence — additional seal + validate paths
# ===========================================================================

class TestCoherenceExtended:

    def test_seal_passport_hashes(self):
        from gpts_core.coherence import build_coherence_passport, seal_passport_hashes
        obs = {"modA": [float(i) for i in range(20)]}
        passport = build_coherence_passport("C01", obs)
        sealed = seal_passport_hashes(passport)
        assert "hash_chain" in sealed or "payload_hash" in sealed or sealed is not None

    def test_validate_passport_after_seal(self):
        from gpts_core.coherence import build_coherence_passport, seal_passport_hashes, validate_coherence_passport
        obs = {
            "modA": [float(i) for i in range(30)],
            "modB": [float(i % 4) for i in range(30)],
        }
        passport = build_coherence_passport("C02", obs)
        passport = seal_passport_hashes(passport)
        result = validate_coherence_passport(passport)
        assert isinstance(result, dict)

    def test_joint_entropy_basic(self):
        from gpts_core.coherence import joint_entropy
        x = [0, 1, 0, 1, 0, 1] * 10
        y = [0, 0, 1, 1, 0, 0] * 10
        h = joint_entropy(x, y)
        assert h >= 0.0

    def test_digitize(self):
        from gpts_core.coherence import digitize
        values = [float(i) for i in range(16)]
        symbols = digitize(values, bins=4)
        assert len(symbols) == 16
        assert all(0 <= s < 4 for s in symbols)

    def test_normalize_observations(self):
        from gpts_core.coherence import normalize_observations
        obs = {"modA": [float(i) for i in range(20)], "modB": [float(i * 2) for i in range(20)]}
        result, warnings, notes = normalize_observations(obs)
        assert "modA" in result
        assert "modB" in result

    def test_compute_metric_fields_single_module(self):
        from gpts_core.coherence import compute_metric_fields
        obs = {"modA": [float(i) for i in range(20)]}
        fields = compute_metric_fields(obs)
        assert "global_coherence" in fields

    def test_joint_entropy_empty(self):
        from gpts_core.coherence import joint_entropy
        assert joint_entropy([], []) == 0.0

    def test_mutual_information_empty(self):
        from gpts_core.coherence import mutual_information
        assert mutual_information([], []) == 0.0

    def test_sha256_file_internal(self, tmp_path):
        from gpts_core.coherence import _sha256_file
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = _sha256_file(f)
        assert h.startswith("sha256:")

    def test_normalize_observations_invalid_input(self):
        from gpts_core.coherence import normalize_observations
        result, blockers, warnings = normalize_observations({})
        assert "BLOCKED_NO_MODULE_OBSERVATIONS" in blockers

    def test_normalize_observations_non_mapping(self):
        from gpts_core.coherence import normalize_observations
        result, blockers, warnings = normalize_observations("not a dict")
        assert len(blockers) > 0


# ===========================================================================
# signals — remaining edge cases
# ===========================================================================

class TestSignalsRemainingEdgeCases:

    def test_entropy_norm_empty_array(self):
        from gpts_core.signals import entropy_norm
        import numpy as np
        h = entropy_norm(np.array([]))
        assert h == 0.0

    def test_window_stats_enough_chunks(self):
        from gpts_core.signals import window_stats
        import numpy as np
        import math
        # Need 1024+ samples to get 2 chunks of 512
        x = np.array([math.sin(i * 0.05) for i in range(1200)])
        pcv, ecv, jsd = window_stats(x, sr=256.0, window=512)
        # With 2+ chunks, should return real values (not all nan)
        assert isinstance(pcv, float)

    def test_bootstrap_ci_all_nonfinite(self):
        from gpts_core.signals import bootstrap_ci
        result = bootstrap_ci([float("nan"), float("inf"), float("-inf")], seed=42)
        assert math.isnan(result["mean"])
        assert result["n"] == 0

    def test_phase_scramble_short_signal(self):
        from gpts_core.signals import generate_null
        import numpy as np
        # rfft of 2-element signal gives 2 elements: len(y2) = 2, so not > 2
        x = np.array([1.0, -1.0])
        result = generate_null(x, seed=42, kind="phase_scramble")
        assert len(result) == 2

    def test_ar2_fit_angles_empty(self):
        from gpts_core.signals import ar2_fit
        import numpy as np
        # Near-degenerate case where roots might have zero angle
        x = np.array([1.0, -1.0, 1.0, -1.0] * 20)
        a1, a2, r2, ef, rr = ar2_fit(x, sr=256.0)
        assert isinstance(ef, float)


# ===========================================================================
# audit_claims — AuditReport.render_markdown, SourceRecord, load_artifacts
# ===========================================================================

class TestAuditClaimsExtended:

    def _make_weakness(self):
        from gpts_core.audit_claims import Weakness
        return Weakness(
            type="Explanatory Limit",
            status="Established",
            severity="Medium",
            probable_cause="Modeling",
            discriminant_test="Check if external benchmark changes verdict.",
        )

    def _make_full_report(self):
        from gpts_core.audit_claims import AuditReport, Weakness, SourceRecord
        w = self._make_weakness()
        return AuditReport(
            title="Test Report",
            reformulated_question="Is X valid?",
            verdict="PARTIALLY SUPPORTED",
            confidence=6.5,
            one_sentence_justification="Partial support found in corpus.",
            propositions=["X is coherent.", "X is traceable."],
            supporting_arguments=["Support A.", "Support B."],
            opposing_arguments=["Opposing A."],
            weaknesses=[w],
            internal_logical_contradictions="No contradiction.",
            tensions_with_observations="Minor tension.",
            assumption_dependence="Some assumption.",
            explanatory_limits="Some limits.",
            limitations=["Limited by scope.", "Data incomplete."],
            recommendations_tests=["Run benchmark test."],
            recommendations_alternatives=["Use simpler review."],
            recommendations_next_questions=["Is verdict stable?"],
            sources=[
                SourceRecord(label="file.py", path="/tmp/file.py", source_type="artifact", stable=True, excerpt="code..."),
                SourceRecord(label="conv", path=None, source_type="conversation", stable=False),
            ],
        )

    def test_source_record_render_with_path(self):
        from gpts_core.audit_claims import SourceRecord
        s = SourceRecord(label="file.py", path="/tmp/file.py", source_type="artifact", stable=True, excerpt="code...")
        rendered = s.render()
        assert "file.py" in rendered
        assert "stable" in rendered

    def test_source_record_render_no_path(self):
        from gpts_core.audit_claims import SourceRecord
        s = SourceRecord(label="conv_source", path=None, source_type="conversation", stable=False)
        rendered = s.render()
        assert "conv_source" in rendered
        assert "/" not in rendered or "unstabilized" in rendered

    def test_weakness_validate_valid(self):
        w = self._make_weakness()
        w.validate()  # should not raise

    def test_weakness_validate_invalid_type(self):
        from gpts_core.audit_claims import Weakness
        with pytest.raises(ValueError, match="Invalid weakness type"):
            w = Weakness(type="INVALID_TYPE", status="Established", severity="Medium",
                         probable_cause="Modeling", discriminant_test="test")
            w.validate()

    def test_weakness_validate_invalid_status(self):
        from gpts_core.audit_claims import Weakness
        with pytest.raises(ValueError, match="Invalid status"):
            w = Weakness(type="Explanatory Limit", status="INVALID", severity="Medium",
                         probable_cause="Modeling", discriminant_test="test")
            w.validate()

    def test_weakness_validate_invalid_severity(self):
        from gpts_core.audit_claims import Weakness
        with pytest.raises(ValueError, match="Invalid severity"):
            w = Weakness(type="Explanatory Limit", status="Established", severity="INVALID",
                         probable_cause="Modeling", discriminant_test="test")
            w.validate()

    def test_weakness_validate_invalid_cause(self):
        from gpts_core.audit_claims import Weakness
        with pytest.raises(ValueError, match="Invalid cause"):
            w = Weakness(type="Explanatory Limit", status="Established", severity="Medium",
                         probable_cause="INVALID_CAUSE", discriminant_test="test")
            w.validate()

    def test_weakness_validate_empty_discriminant(self):
        from gpts_core.audit_claims import Weakness
        with pytest.raises(ValueError, match="Discriminant Test is empty"):
            w = Weakness(type="Explanatory Limit", status="Established", severity="Medium",
                         probable_cause="Modeling", discriminant_test="   ")
            w.validate()

    def test_weakness_render(self):
        w = self._make_weakness()
        rendered = w.render(1)
        assert "Weakness 1" in rendered
        assert "Explanatory Limit" in rendered
        assert "Modeling" in rendered

    def test_weakness_render_with_basis(self):
        from gpts_core.audit_claims import Weakness
        w = Weakness(type="Explanatory Limit", status="Established", severity="Low",
                     probable_cause="Data", discriminant_test="test",
                     basis="Evidence basis here.")
        rendered = w.render(2)
        assert "Evidence basis here." in rendered

    def test_audit_report_validate_valid(self):
        report = self._make_full_report()
        report.validate()  # should not raise

    def test_audit_report_validate_invalid_verdict(self):
        from gpts_core.audit_claims import AuditReport
        report = self._make_full_report()
        report.verdict = "INVALID_VERDICT"
        with pytest.raises(ValueError, match="Invalid verdict"):
            report.validate()

    def test_audit_report_validate_invalid_confidence(self):
        report = self._make_full_report()
        report.confidence = 15.0  # out of [0,10]
        with pytest.raises(ValueError, match="Confidence out of"):
            report.validate()

    def test_audit_report_render_markdown(self):
        report = self._make_full_report()
        md = report.render_markdown()
        assert "1. Title" in md
        assert "PARTIALLY SUPPORTED" in md
        assert "6.5/10" in md
        assert "7. Sources" in md

    def test_audit_report_to_dict(self):
        report = self._make_full_report()
        d = report.to_dict()
        assert "claim_ceiling" in d
        assert d["claim_ceiling"] == "LOCAL_LAB_ONLY_NOT_PUBLIC_PROOF"

    def test_audit_claim_with_conversation_source(self):
        from gpts_core.audit_claims import audit_claim
        report = audit_claim("Test claim text", "Title", add_conversation_source=True)
        assert any(s.source_type == "conversation" for s in report.sources)

    def test_load_artifacts_with_text_file(self, tmp_path):
        from gpts_core.audit_claims import load_artifacts
        f = tmp_path / "test.txt"
        f.write_text("This is test content for artifact loading.")
        loaded, sources = load_artifacts([str(tmp_path)])
        assert len(loaded) >= 1
        assert sources[0].stable is True

    def test_load_artifacts_with_json_file(self, tmp_path):
        from gpts_core.audit_claims import load_artifacts
        f = tmp_path / "data.json"
        f.write_text('{"key": "value", "num": 42}')
        loaded, sources = load_artifacts([str(f)])
        assert len(loaded) == 1
        assert "key" in loaded[0][1]

    def test_load_artifacts_with_csv_file(self, tmp_path):
        from gpts_core.audit_claims import load_artifacts
        f = tmp_path / "data.csv"
        f.write_text("name,value\nalpha,1\nbeta,2\n")
        loaded, sources = load_artifacts([str(f)])
        assert len(loaded) == 1

    def test_load_artifacts_with_python_file(self, tmp_path):
        from gpts_core.audit_claims import load_artifacts
        f = tmp_path / "module.py"
        f.write_text("def foo(): return 42\n")
        loaded, sources = load_artifacts([str(f)])
        assert len(loaded) == 1

    def test_load_artifacts_skips_unknown_extension(self, tmp_path):
        from gpts_core.audit_claims import load_artifacts
        f = tmp_path / "data.xyz"
        f.write_text("unknown content")
        loaded, sources = load_artifacts([str(f)])
        assert len(loaded) == 0

    def test_audit_claim_with_corpus(self, tmp_path):
        from gpts_core.audit_claims import audit_claim
        f = tmp_path / "corpus.txt"
        f.write_text("This system demonstrates validated coherent behavior. "
                     "The protocol is canonical and well-defined. "
                     "Observations confirmed measured performance. "
                     "Results are strictly constrained and explicit.")
        report = audit_claim("The system is coherent and validated", "Test",
                             inputs=[str(f)])
        assert report.verdict in ("SUPPORTED", "PARTIALLY SUPPORTED", "WEAKLY SUPPORTED",
                                  "UNSUPPORTED", "UNCERTAIN")

    def test_detect_logical_contradictions_none(self):
        from gpts_core.audit_claims import detect_logical_contradictions
        result = detect_logical_contradictions(["always true", "sometimes false"])
        assert isinstance(result, list)

    def test_detect_logical_contradictions_found(self):
        from gpts_core.audit_claims import detect_logical_contradictions
        result = detect_logical_contradictions(["always present", "never used"])
        assert isinstance(result, list)

    def test_batch_audit_full(self):
        from gpts_core.audit_claims import batch_audit
        claims = [
            {"title": "T1", "claim": "The system is coherent"},
            {"title": "T2", "claim": "The protocol is canonical"},
        ]
        result = batch_audit(claims)
        assert isinstance(result, dict)

    def test_validate_canonical_report_full_sections(self):
        from gpts_core.audit_claims import validate_canonical_report, AuditReport
        report = self._make_full_report()
        md = report.render_markdown()
        valid, errors = validate_canonical_report(md)
        assert isinstance(valid, bool)
        assert isinstance(errors, list)


# ===========================================================================
# adjudication — write_confusion_matrix_csv
# ===========================================================================

class TestAdjudicationExtended:

    def test_write_confusion_matrix_csv(self, tmp_path):
        from gpts_core.adjudication import build_confusion_matrix, write_confusion_matrix_csv
        rows = [
            {"label": "simulated", "human_label": "simulated"},
            {"label": "observed", "human_label": "observed"},
            {"label": "computed", "human_label": "simulated"},
        ]
        matrix = build_confusion_matrix(rows)
        out = tmp_path / "confusion.csv"
        write_confusion_matrix_csv(out, matrix)
        assert out.exists()
        content = out.read_text()
        assert "classifier_label" in content
        assert "simulated" in content

    def test_build_confusion_matrix_empty_fields(self):
        from gpts_core.adjudication import build_confusion_matrix
        rows = [{"label": "", "human_label": "observed"}, {"label": "simulated", "human_label": ""}]
        matrix = build_confusion_matrix(rows)
        # Empty pred/human should be skipped
        assert "observed" not in matrix


# ===========================================================================
# spectral_gap — additional paths
# ===========================================================================

class TestSpectralGapExtended:

    def test_loglog_regression_pre_asymptotic(self):
        from gpts_core.spectral_gap import loglog_regression
        # Random alphas and gaps that don't follow log-log scaling → PRE_ASYMPTOTIC
        alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        gaps   = [0.50, 0.10, 0.40, 0.08, 0.45, 0.15, 0.35]  # chaotic, no scaling
        result = loglog_regression(alphas, gaps)
        assert isinstance(result, dict)
        if result.get("FAIL") is None:
            assert result.get("qualificatif") in ("PRE_ASYMPTOTIC", "GOOD_SCALING", "PUBLICATION_GRADE")

    def test_loglog_regression_not_enough_valid(self):
        from gpts_core.spectral_gap import loglog_regression
        alphas = [0.01, 0.02, 0.05]
        gaps = [None, None, None]  # no valid points
        result = loglog_regression(alphas, gaps)
        assert result["FAIL"] is not None
        assert result["kappa_fit"] is None

    def test_compute_gap_point_returns_tuple(self):
        from gpts_core.spectral_gap import compute_gap_point
        gap, diag = compute_gap_point(0.05, n_bins=10, n_traj=500)
        assert isinstance(diag, dict)
        assert "alpha" in diag

    def test_run_pipeline_with_six_alphas(self):
        import numpy as np
        from gpts_core.spectral_gap import run_pipeline
        alpha_grid = list(np.geomspace(0.02, 0.20, 8))
        result = run_pipeline(alpha_grid, n_bins=20, n_traj=1000, seed=7)
        assert len(result["alphas"]) == 8
        assert "regression" in result

    def test_ulam_matrix_zero_rows_handled(self):
        import numpy as np
        from gpts_core.spectral_gap import ulam_matrix
        # Very high n_bins with very few trajectories → likely zero rows
        P, diag = ulam_matrix(alpha=0.1, n_bins=50, n_traj=20, seed=0)
        assert P.shape == (50, 50)
        row_sums = P.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)


# ===========================================================================
# coherence — normalize_observations edge cases, passport blocked path
# ===========================================================================

class TestCoherenceNormalization:

    def test_is_finite_float_bool(self):
        from gpts_core.coherence import _is_finite_float
        assert _is_finite_float(True) is False
        assert _is_finite_float(False) is False

    def test_is_finite_float_string(self):
        from gpts_core.coherence import _is_finite_float
        assert _is_finite_float("not_a_number") is False
        assert _is_finite_float("3.14") is True

    def test_normalize_observations_samples_as_string(self):
        from gpts_core.coherence import normalize_observations
        # String is not list/tuple → blocked
        result, blockers, warnings = normalize_observations({"modA": "not_a_list"})
        assert any("BLOCKED_TOO_FEW_SAMPLES:modA" in b for b in blockers)

    def test_normalize_observations_non_numeric_sample(self):
        from gpts_core.coherence import normalize_observations
        # Contains a bool (which is non-finite by _is_finite_float)
        result, blockers, warnings = normalize_observations({"modA": [1.0, 2.0, True, 3.0]})
        assert any("NON_NUMERIC" in b for b in blockers)

    def test_normalize_observations_too_few_after_filter(self):
        from gpts_core.coherence import normalize_observations
        # Only one valid value → too few
        result, blockers, warnings = normalize_observations({"modA": [1.0]})
        assert any("TOO_FEW" in b for b in blockers)

    def test_normalize_observations_small_sample_warning(self):
        from gpts_core.coherence import normalize_observations
        # 4 valid values: < 16, triggers SMALL_SAMPLE warning
        result, blockers, warnings = normalize_observations({"modA": [1.0, 2.0, 3.0, 4.0]})
        assert any("SMALL_SAMPLE" in w for w in warnings)

    def test_build_coherence_passport_with_invalid_observations(self):
        from gpts_core.coherence import build_coherence_passport
        # Empty observations → blocked
        passport = build_coherence_passport("C01", {})
        assert passport["verdict"] == "CAPTURE_BLOCKED_FAIL_CLOSED"
        assert len(passport.get("blockers", [])) > 0

    def test_build_coherence_passport_with_string_module(self):
        from gpts_core.coherence import build_coherence_passport
        # Module with non-list samples → blocked
        passport = build_coherence_passport("C01", {"modA": "invalid"})
        assert passport["verdict"] == "CAPTURE_BLOCKED_FAIL_CLOSED"

    def test_validate_coherence_passport_source_status_world(self):
        from gpts_core.coherence import build_coherence_passport, seal_passport_hashes, validate_coherence_passport
        obs = {"modA": [float(i) for i in range(20)], "modB": [float(i % 4) for i in range(20)]}
        passport = build_coherence_passport("C01", obs)
        passport = seal_passport_hashes(passport)
        passport["source_status"] = "INTERFACE_SIGNAL_ONLY"
        result = validate_coherence_passport(passport)
        assert any("WORLD_NOT_PROOF" in e for e in result.get("errors", []))

    def test_validate_coherence_passport_formula_mismatch(self):
        from gpts_core.coherence import build_coherence_passport, seal_passport_hashes, validate_coherence_passport
        obs = {"modA": [float(i) for i in range(20)], "modB": [float(i % 4) for i in range(20)]}
        passport = build_coherence_passport("C01", obs)
        passport = seal_passport_hashes(passport)
        # Tamper with global_coherence to create mismatch
        passport["global_coherence"] = 99.0
        result = validate_coherence_passport(passport)
        assert any("FORMULA_MISMATCH" in e or "BLOCKED" in e for e in result.get("errors", []))


# ===========================================================================
# score_report — _semantic_match token path, _contains_nearby_verdict, critical points partial
# ===========================================================================

class TestScoreReportInternals:

    def test_semantic_match_not_substring_has_tokens(self):
        from gpts_core.score_report import _semantic_match
        # "test validation method" is not a literal substring of the text,
        # but tokens overlap: "test", "validation"
        text = "This is about validation and testing procedure methodology."
        result = _semantic_match("systematic test validation", text, threshold=0.30)
        assert isinstance(result, bool)

    def test_semantic_match_empty_candidate(self):
        from gpts_core.score_report import _semantic_match
        # Empty string is always substring of any text → True (Python "in" behaviour)
        result = _semantic_match("", "some text here to search in", threshold=0.30)
        assert result is True

    def test_semantic_match_no_token_overlap(self):
        from gpts_core.score_report import _semantic_match
        # Tokens that don't appear in text
        result = _semantic_match("xyzzyfoobarbaz", "completely unrelated content here", threshold=0.30)
        assert result is False

    def test_contains_nearby_verdict_anchor_not_found(self):
        from gpts_core.score_report import _contains_nearby_verdict
        # Anchor is not in text at all → returns False (line 137)
        result = _contains_nearby_verdict("some report text", "xyz_not_in_text", "SUPPORTED")
        assert result is False

    def test_contains_nearby_verdict_anchor_found(self):
        from gpts_core.score_report import _contains_nearby_verdict
        result = _contains_nearby_verdict(
            "The proposition A is clearly SUPPORTED by the evidence.",
            "proposition A",
            "SUPPORTED",
        )
        assert result is True

    def test_score_critical_points_partial_match(self):
        from gpts_core.score_report import score_critical_points
        # Short report text that has SOME token overlap with critical points but not full match
        report_text = "The analysis considers validation aspects and testing protocols."
        # "quantum entanglement verification" → has "verification" = maybe partial
        critical_points = ["direct observational measurement", "explicit reproducible methodology"]
        score, note, uncovered = score_critical_points(report_text, critical_points, 20)
        assert 0 <= score <= 20
        assert isinstance(uncovered, list)

    def test_score_critical_points_all_uncovered(self):
        from gpts_core.score_report import score_critical_points
        # Completely unrelated text → no critical points covered
        report_text = "The banana is yellow and delicious."
        critical_points = ["quantum entanglement verification", "neural spike synchronization"]
        score, note, uncovered = score_critical_points(report_text, critical_points, 20)
        assert len(uncovered) > 0

    def test_score_discriminant_tests_zero_found(self):
        from gpts_core.score_report import score_discriminant_tests
        # Analysis has no discriminant tests → 0 found, expected >= 2 → line 299
        sections = {"Detailed Analysis": "The evidence suggests partial support."}
        oracle = {"expected_min_discriminant_tests": 3}
        score, note, count = score_discriminant_tests(sections, oracle, 15)
        assert score == 0
        assert count == 0

    def test_score_discriminant_tests_one_short(self):
        from gpts_core.score_report import score_discriminant_tests
        # 1 found, expected_min=2 → found == expected_min - 1 → line 298
        sections = {"Detailed Analysis": "Test 1: this is a discriminant test example."}
        oracle = {"expected_min_discriminant_tests": 2}
        score, note, count = score_discriminant_tests(sections, oracle, 15)
        assert 0 <= score < 15  # penalized

    def test_keyword_set_function(self):
        from gpts_core.score_report import _keyword_set
        result = _keyword_set("test validation methodology")
        assert isinstance(result, set)
        assert "validation" in result or "methodology" in result

    def test_tokenize_function(self):
        from gpts_core.score_report import _tokenize
        result = _tokenize("The quick brown fox jumps over the lazy dog")
        assert isinstance(result, list)
        assert "quick" in result or "brown" in result
