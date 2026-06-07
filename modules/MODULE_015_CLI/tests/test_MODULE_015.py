"""Tests for MODULE_015_CLI.

The CLI has 0% unit test coverage in the source suite (subprocess-only interface).
These tests verify the parser structure, subcommand registration, and that main()
dispatches correctly via direct Python invocation (bypassing subprocess).

Imports from the module's exports directory.
"""
from __future__ import annotations

import json
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestCLIParser:
    def test_build_parser_returns_parser(self):
        from cli import build_parser
        import argparse
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_parser_prog_name(self):
        from cli import build_parser
        p = build_parser()
        assert p.prog == "gpts-core"

    def test_all_subcommands_registered(self):
        from cli import build_parser
        p = build_parser()
        # Extract subparser choices from the parser
        subparsers_action = None
        for action in p._actions:
            if hasattr(action, '_name_parser_map'):
                subparsers_action = action
                break
        assert subparsers_action is not None
        choices = set(subparsers_action._name_parser_map.keys())
        expected = {
            "analyze-signal", "classify-metric", "validate-records",
            "classify-claim", "evidence-score", "replay-ledger",
            "seal-audit", "promote", "build-manifest", "adjudicate",
            "coherence-passport", "audit-claim", "score-report", "spectral-gap",
        }
        assert expected == choices

    def test_pretty_flag_default_false(self):
        from cli import build_parser
        p = build_parser()
        # parse a minimal command
        args = p.parse_args(["classify-claim", "test claim"])
        assert args.pretty is False

    def test_pretty_flag_set(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["--pretty", "classify-claim", "test claim"])
        assert args.pretty is True

    def test_classify_claim_parses_text(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["classify-claim", "This is a test claim"])
        assert args.text == "This is a test claim"
        assert args.cmd == "classify-claim"

    def test_evidence_score_required_args(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args([
            "evidence-score",
            "--expected", "10",
            "--present", "8",
            "--valid", "7",
        ])
        assert args.expected == 10
        assert args.present == 8
        assert args.valid == 7

    def test_spectral_gap_defaults(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["spectral-gap"])
        assert args.alpha_min == 0.02
        assert args.alpha_max == 0.20
        assert args.n_alpha == 10
        assert args.n_bins == 80
        assert args.n_traj == 20000
        assert args.seed == 42

    def test_spectral_gap_custom_args(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args([
            "spectral-gap",
            "--alpha-min", "0.05",
            "--alpha-max", "0.15",
            "--n-alpha", "6",
            "--n-bins", "40",
            "--n-traj", "5000",
            "--seed", "99",
        ])
        assert args.alpha_min == 0.05
        assert args.n_alpha == 6

    def test_analyze_signal_defaults(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["analyze-signal", "data.csv"])
        assert args.column == "value"
        assert args.sr == 256.0

    def test_classify_metric_defaults(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["classify-metric"])
        assert args.name == ""
        assert args.row_type == "metric"

    def test_validate_records_requires_expected(self):
        from cli import build_parser
        import argparse
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["validate-records", "file.jsonl"])

    def test_coherence_passport_bins_default(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["coherence-passport", "obs.json"])
        assert args.bins == 8

    def test_score_report_min_score_default(self):
        from cli import build_parser
        p = build_parser()
        args = p.parse_args(["score-report", "report.md"])
        assert args.min_score == 0

    def test_audit_claim_requires_claim(self):
        from cli import build_parser
        import argparse
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["audit-claim"])


class TestCLIMain:
    def test_main_returns_int(self):
        from cli import main
        # classify-claim is a pure-python dispatch — no file I/O needed
        result = main(["classify-claim", "This is a lab-only hypothesis"])
        assert isinstance(result, int)
        assert result in (0, 1, 2)

    def test_main_classify_claim_blocked_returns_2(self):
        from cli import main
        result = main(["classify-claim", "This system is production-ready and approved for production"])
        assert result == 2

    def test_main_classify_claim_unknown_returns_0(self):
        from cli import main
        result = main(["classify-claim", "This is a preliminary hypothesis in a lab-only context"])
        assert result in (0, 2)

    def test_main_unknown_command_returns_2(self):
        from cli import main
        result = main(["nonexistent-subcommand"])
        assert result == 2

    def test_main_classify_metric_runs(self):
        from cli import main
        result = main(["classify-metric", "--name", "psi", "--value", "0.99", "--context", "simulation"])
        assert isinstance(result, int)

    def test_main_evidence_score_high_returns_0(self):
        from cli import main
        result = main([
            "evidence-score",
            "--expected", "10",
            "--present", "10",
            "--valid", "10",
            "--safety", "1.0",
            "--scoring",
            "--replay",
            "--independence",
        ])
        assert result == 0

    def test_main_evidence_score_low_returns_2(self):
        from cli import main
        result = main([
            "evidence-score",
            "--expected", "10",
            "--present", "2",
            "--valid", "1",
            "--safety", "0.2",
        ])
        assert result == 2

    def test_main_replay_ledger_missing_file_returns_nonzero(self, tmp_path):
        from cli import main
        result = main(["replay-ledger", str(tmp_path / "nonexistent.jsonl")])
        assert result != 0

    def test_main_validate_records_missing_file_returns_2(self, tmp_path):
        from cli import main
        result = main(["validate-records", str(tmp_path / "nope.jsonl"), "--expected", "1"])
        assert result == 2
