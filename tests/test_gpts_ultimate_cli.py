"""Tests for gpts_ultimate.cli — all 14 subcommands via argparse."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gpts_ultimate.cli import build_parser, main


SINE = np.sin(np.linspace(0, 8 * np.pi, 512))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_signal_csv(path: Path, values=None, col="value"):
    if values is None:
        values = SINE
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([col])
        for v in values:
            w.writerow([v])


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------
class TestParser:
    def test_parser_built(self):
        p = build_parser()
        assert p is not None

    def test_all_subcommands(self):
        p = build_parser()
        subparsers_action = next(
            a for a in p._actions if hasattr(a, "_name_parser_map")
        )
        commands = set(subparsers_action._name_parser_map.keys())
        expected = {
            "analyze-signal", "classify-metric", "validate-records",
            "classify-claim", "evidence-score", "replay-ledger",
            "seal-audit", "promote", "build-manifest", "adjudicate",
            "lorenz", "berzerker", "math-metrics", "run-pipeline",
        }
        assert expected <= commands


# ---------------------------------------------------------------------------
# analyze-signal
# ---------------------------------------------------------------------------
class TestCmdAnalyzeSignal:
    def test_basic(self, tmp_path, capsys):
        p = tmp_path / "sig.csv"
        write_signal_csv(p)
        rc = main(["analyze-signal", str(p)])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "structure_score" in out

    def test_missing_column(self, tmp_path, capsys):
        p = tmp_path / "sig.csv"
        write_signal_csv(p)
        rc = main(["analyze-signal", str(p), "--column", "nonexistent"])
        assert rc == 2

    def test_pretty(self, tmp_path, capsys):
        p = tmp_path / "sig.csv"
        write_signal_csv(p)
        main(["--pretty", "analyze-signal", str(p)])
        out = capsys.readouterr().out
        assert "\n" in out  # indented JSON


# ---------------------------------------------------------------------------
# classify-metric
# ---------------------------------------------------------------------------
class TestCmdClassifyMetric:
    def test_basic(self, capsys):
        rc = main(["classify-metric", "--name", "accuracy", "--value", "0.92"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "label" in out

    def test_simulation_label(self, capsys):
        main(["classify-metric", "--name", "lyapunov", "--value", "0.5",
              "--context", "simulated trajectory"])
        out = json.loads(capsys.readouterr().out)
        assert out["label"] in ("simulated", "computed", "observed", "reported",
                                "symbolic", "unsupported")


# ---------------------------------------------------------------------------
# classify-claim
# ---------------------------------------------------------------------------
class TestCmdClassifyClaim:
    def test_blocked_returns_2(self):
        rc = main(["classify-claim", "production-ready system beats GPT"])
        assert rc == 2

    def test_bounded_returns_0(self, capsys):
        rc = main(["classify-claim", "this is a local simulation prototype"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["status"] == "ALLOWED_BOUNDED"


# ---------------------------------------------------------------------------
# evidence-score
# ---------------------------------------------------------------------------
class TestCmdEvidenceScore:
    def test_blocked(self):
        rc = main(["evidence-score", "--expected", "10", "--present", "10", "--valid", "0"])
        assert rc == 2

    def test_full_evidence(self, capsys):
        rc = main(["evidence-score", "--expected", "10", "--present", "10", "--valid", "10",
                   "--safety", "1.0", "--scoring", "--replay", "--independence"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["public_claim_allowed"] is True


# ---------------------------------------------------------------------------
# lorenz
# ---------------------------------------------------------------------------
class TestCmdLorenz:
    def test_metadata_output(self, capsys):
        rc = main(["lorenz", "--nodes", "3", "--t-max", "50", "--discard", "5", "--seed", "0"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["nodes"] == 3

    def test_csv_output(self, tmp_path):
        out_path = tmp_path / "ensemble.csv"
        rc = main(["lorenz", "--nodes", "2", "--t-max", "50", "--discard", "5",
                   "--seed", "0", "--out", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        with out_path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["node_0", "node_1"]


# ---------------------------------------------------------------------------
# berzerker
# ---------------------------------------------------------------------------
class TestCmdBerzerker:
    def test_returns_report(self, capsys):
        rc = main(["berzerker", "--seed-start", "5", "--n-seeds", "8",
                   "--coupling-seed", "0"])
        out = json.loads(capsys.readouterr().out)
        assert "verdict" in out

    def test_output_file(self, tmp_path, capsys):
        out_path = str(tmp_path / "report.json")
        main(["berzerker", "--seed-start", "5", "--n-seeds", "8",
              "--coupling-seed", "0", "--out", out_path])
        with open(out_path) as f:
            loaded = json.load(f)
        assert "verdict" in loaded


# ---------------------------------------------------------------------------
# math-metrics
# ---------------------------------------------------------------------------
class TestCmdMathMetrics:
    def test_lookup(self, capsys):
        rc = main(["math-metrics", "BRIER_SCORE"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "formula" in out or "name" in out

    def test_validate_pass(self, capsys):
        rc = main(["math-metrics", "BRIER_SCORE", "--result", "0.05", "--r2", "0.97"])
        out = json.loads(capsys.readouterr().out)
        assert out["verdict"]["status"] == "PASS"

    def test_invalid_formula(self, capsys):
        rc = main(["math-metrics", "NONEXISTENT_FORMULA"])
        assert rc == 2


# ---------------------------------------------------------------------------
# run-pipeline
# ---------------------------------------------------------------------------
class TestCmdRunPipeline:
    def test_basic(self, tmp_path, capsys):
        p = tmp_path / "traj.csv"
        write_signal_csv(p)
        rc = main(["run-pipeline", str(p), "--n-seeds", "7", "--n-null", "5"])
        out = json.loads(capsys.readouterr().out)
        assert "tribunal" in out
        assert "final_verdict" in out["tribunal"]

    def test_with_ledger(self, tmp_path, capsys):
        p = tmp_path / "traj.csv"
        write_signal_csv(p)
        ledger_path = str(tmp_path / "audit.jsonl")
        main(["run-pipeline", str(p), "--n-seeds", "7", "--n-null", "5",
              "--ledger", ledger_path])
        assert Path(ledger_path).exists()


# ---------------------------------------------------------------------------
# build-manifest
# ---------------------------------------------------------------------------
class TestCmdBuildManifest:
    def test_basic(self, tmp_path, capsys):
        root = tmp_path / "src"
        root.mkdir()
        (root / "file.txt").write_text("hello")
        out_dir = tmp_path / "manifest_out"
        out_dir.mkdir()
        rc = main(["build-manifest", str(root), "--out", str(out_dir)])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["file_count"] >= 1
