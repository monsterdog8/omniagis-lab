"""Tests for gpts_core.cli — unified CLI subcommand handlers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from gpts_core.cli import (
    build_parser,
    cmd_classify_metric,
    cmd_classify_claim,
    cmd_validate_records,
    cmd_analyze_signal,
    cmd_replay_ledger,
    cmd_evidence_score,
    cmd_adjudicate,
    cmd_build_manifest,
    cmd_seal_audit,
    cmd_promote,
    main,
)
from gpts_core.evidence import build_raw_record


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_has_all_subcommands(self):
        p = build_parser()
        subparsers_action = next(
            a for a in p._actions if hasattr(a, "_name_parser_map")
        )
        cmds = set(subparsers_action._name_parser_map.keys())
        expected = {
            "analyze-signal", "classify-metric", "validate-records",
            "classify-claim", "evidence-score", "replay-ledger",
            "seal-audit", "promote", "build-manifest", "adjudicate",
        }
        assert expected.issubset(cmds)

    def test_pretty_flag_global(self):
        p = build_parser()
        args = p.parse_args(["--pretty", "classify-claim", "hello"])
        assert args.pretty is True

    def test_no_pretty_default(self):
        p = build_parser()
        args = p.parse_args(["classify-claim", "hello"])
        assert args.pretty is False


# ---------------------------------------------------------------------------
# cmd_classify_claim
# ---------------------------------------------------------------------------

class TestCmdClassifyClaim:
    def _args(self, text, pretty=False):
        ns = argparse.Namespace(text=text, pretty=pretty)
        return ns

    def test_unknown_text_returns_zero(self, capsys):
        code = cmd_classify_claim(self._args("the weather is nice"))
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["status"] != "BLOCKED"

    def test_blocked_text_returns_two(self, capsys):
        code = cmd_classify_claim(self._args("achieves SOTA on all benchmarks"))
        assert code == 2

    def test_pretty_outputs_indented(self, capsys):
        cmd_classify_claim(self._args("hello", pretty=True))
        out = capsys.readouterr().out
        assert "\n" in out


# ---------------------------------------------------------------------------
# cmd_classify_metric
# ---------------------------------------------------------------------------

class TestCmdClassifyMetric:
    def _args(self, name="mean_score", value="0.85", context="", row_type="metric", pretty=False):
        return argparse.Namespace(name=name, value=value, context=context,
                                   row_type=row_type, pretty=pretty)

    def test_returns_zero(self, capsys):
        code = cmd_classify_metric(self._args())
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert "label" in out
        assert "confidence" in out


# ---------------------------------------------------------------------------
# cmd_validate_records
# ---------------------------------------------------------------------------

class TestCmdValidateRecords:
    def _write_jsonl(self, path, records):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_valid_records_returns_zero(self, tmp_path, capsys):
        records = [build_raw_record("r", "t", f"o{i}", "m", f"text{i}", token_count=10, latency_ms=5)
                   for i in range(2)]
        p = tmp_path / "records.jsonl"
        self._write_jsonl(p, records)
        args = argparse.Namespace(file=str(p), expected=2, run_id="run1",
                                   no_strict=False, pretty=False)
        code = cmd_validate_records(args)
        assert code == 0

    def test_wrong_count_returns_nonzero(self, tmp_path, capsys):
        records = [build_raw_record("r", "t", "o1", "m", "text", token_count=5, latency_ms=10)]
        p = tmp_path / "records.jsonl"
        self._write_jsonl(p, records)
        args = argparse.Namespace(file=str(p), expected=5, run_id="", no_strict=False, pretty=False)
        code = cmd_validate_records(args)
        assert code == 2


# ---------------------------------------------------------------------------
# cmd_analyze_signal
# ---------------------------------------------------------------------------

class TestCmdAnalyzeSignal:
    def _write_csv(self, path, col, values):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[col])
            w.writeheader()
            for v in values:
                w.writerow({col: v})

    def test_valid_signal(self, tmp_path, capsys):
        p = tmp_path / "signal.csv"
        import numpy as np
        vals = np.sin(np.linspace(0, 4 * np.pi, 512)).tolist()
        self._write_csv(p, "value", vals)
        args = argparse.Namespace(file=str(p), column="value", sr=256.0, pretty=False)
        code = cmd_analyze_signal(args)
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert "structure_score" in out

    def test_empty_column_returns_two(self, tmp_path, capsys):
        p = tmp_path / "signal.csv"
        self._write_csv(p, "value", ["not_a_number", "also_not"])
        args = argparse.Namespace(file=str(p), column="value", sr=256.0, pretty=False)
        code = cmd_analyze_signal(args)
        assert code == 2
        assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# cmd_replay_ledger
# ---------------------------------------------------------------------------

class TestCmdReplayLedger:
    def test_missing_file_returns_nonzero(self, tmp_path, capsys):
        args = argparse.Namespace(file=str(tmp_path / "missing.jsonl"), pretty=False)
        code = cmd_replay_ledger(args)
        assert code == 1

    def test_valid_ledger_returns_zero(self, tmp_path, capsys):
        import hashlib
        event = {"event_id": "000001", "timestamp": "2024-01-01T00:00:00Z",
                 "type": "test", "data": {}, "signature": "sig"}
        p = tmp_path / "ledger.jsonl"
        p.write_text(json.dumps(event), encoding="utf-8")
        args = argparse.Namespace(file=str(p), pretty=False)
        code = cmd_replay_ledger(args)
        assert code == 0


# ---------------------------------------------------------------------------
# cmd_evidence_score
# ---------------------------------------------------------------------------

class TestCmdEvidenceScore:
    def test_no_evidence_returns_two(self, capsys):
        args = argparse.Namespace(
            expected=0, present=0, valid=0, sidecars=0,
            safety=0.5, scoring=False, replay=False, independence=False,
            pretty=False,
        )
        code = cmd_evidence_score(args)
        assert code == 2


# ---------------------------------------------------------------------------
# cmd_adjudicate
# ---------------------------------------------------------------------------

class TestCmdAdjudicate:
    def test_basic_csv(self, tmp_path, capsys):
        p = tmp_path / "adj.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "human_label"])
            w.writeheader()
            w.writerow({"label": "computed", "human_label": "computed"})
        args = argparse.Namespace(file=str(p), pretty=False)
        code = cmd_adjudicate(args)
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["verdict"] == "ADJUDICATION_SCORED_LOCAL_ONLY"


# ---------------------------------------------------------------------------
# cmd_build_manifest
# ---------------------------------------------------------------------------

class TestCmdBuildManifest:
    def test_builds_manifest(self, tmp_path, capsys):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f.py").write_text("pass", encoding="utf-8")
        out = tmp_path / "out"
        args = argparse.Namespace(root=str(src), out=str(out), label="manifest", pretty=False)
        code = cmd_build_manifest(args)
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert "aggregate_hash" in result


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

class TestCmdSealAudit:
    def test_empty_ledger_all_valid(self, tmp_path, capsys):
        ledger = tmp_path / "seal.jsonl"
        ledger.write_text("", encoding="utf-8")
        args = argparse.Namespace(ledger=str(ledger), root=str(tmp_path), pretty=False)
        code = cmd_seal_audit(args)
        assert code == 0


class TestCmdPromote:
    def _write_matrix(self, path):
        import csv
        row = {
            "provenance_chain": "COMPLETE",
            "dependency_resolution_status": "RESOLVED",
            "raw_ledger_presence": "PRESENT_JSONL",
            "replay_status": "REPLAY_PASS",
            "verdict": "VALIDATED",
            "claim_ceiling": "INTERNAL_ONLY",
        }
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    def test_with_candidates(self, tmp_path, capsys):
        p = tmp_path / "matrix.csv"
        self._write_matrix(p)
        args = argparse.Namespace(matrices=[str(p)], pretty=False)
        code = cmd_promote(args)
        assert code == 0

    def test_no_candidates_returns_one(self, tmp_path, capsys):
        p = tmp_path / "matrix.csv"
        row = {
            "provenance_chain": "COMPLETE",
            "dependency_resolution_status": "RESOLVED",
            "raw_ledger_presence": "PRESENT_JSONL",
            "replay_status": "UNKNOWN",
            "verdict": "VALIDATED",
            "claim_ceiling": "LOCAL_ONLY",
        }
        import csv
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        args = argparse.Namespace(matrices=[str(p)], pretty=False)
        code = cmd_promote(args)
        assert code == 1


class TestMain:
    def test_classify_claim_via_main(self, capsys):
        code = main(["classify-claim", "the weather is nice"])
        assert code == 0

    def test_classify_metric_via_main(self, capsys):
        code = main(["classify-metric", "--name", "mean_score", "--value", "0.85"])
        assert code == 0

    def test_error_returns_nonzero(self, capsys):
        code = main(["replay-ledger", "/nonexistent/path/ledger.jsonl"])
        assert code != 0
