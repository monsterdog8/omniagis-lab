"""Tests for omniagis.cli — main(), _verdict_exit_code(), and JSON/text output."""
from __future__ import annotations

import json
import sys

import pytest

from omniagis.cli import _verdict_exit_code, main


# ---------------------------------------------------------------------------
# _verdict_exit_code
# ---------------------------------------------------------------------------

class TestVerdictExitCode:
    def test_pass_is_0(self):
        assert _verdict_exit_code("PASS") == 0

    def test_partial_pass_is_1(self):
        assert _verdict_exit_code("PARTIAL PASS") == 1

    def test_no_pass_is_2(self):
        assert _verdict_exit_code("NO PASS") == 2

    def test_unknown_verdict_is_2(self):
        assert _verdict_exit_code("ANYTHING_ELSE") == 2

    def test_empty_string_is_2(self):
        assert _verdict_exit_code("") == 2


# ---------------------------------------------------------------------------
# main() — text output (default)
# ---------------------------------------------------------------------------

class TestMainTextOutput:
    def test_exits_with_int_code(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
        assert exc.value.code in (0, 1, 2)

    def test_text_output_contains_sections(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "MODE MAVERICK" in out

    def test_text_output_contains_global_verdict(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "GLOBAL VERDICT" in out

    def test_text_output_contains_target_path(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path)])
        out = capsys.readouterr().out
        assert str(tmp_path) in out

    def test_text_is_default_format(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path)])
        out = capsys.readouterr().out
        # text output starts with the MAVERICK banner, not a JSON brace
        assert not out.strip().startswith("{")

    def test_nonempty_dir_with_py_files(self, tmp_path, capsys):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n")
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
        assert exc.value.code in (0, 1, 2)
        out = capsys.readouterr().out
        assert "GLOBAL VERDICT" in out


# ---------------------------------------------------------------------------
# main() — JSON output
# ---------------------------------------------------------------------------

class TestMainJsonOutput:
    def test_json_output_is_valid_json(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_json_output_has_required_keys(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        for key in ("path", "global_verdict", "scorecard", "file_audit",
                    "tensions", "cleanup_plan", "validation_plan", "minimal_core"):
            assert key in data, f"missing key: {key}"

    def test_json_path_matches_input(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["path"] == str(tmp_path)

    def test_json_global_verdict_is_valid(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["global_verdict"] in ("PASS", "PARTIAL PASS", "NO PASS")

    def test_json_exit_code_matches_verdict(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        expected = _verdict_exit_code(data["global_verdict"])
        assert exc.value.code == expected

    def test_json_with_python_files(self, tmp_path, capsys):
        (tmp_path / "a.py").write_text("def bar():\n    pass\n")
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "global_verdict" in data


# ---------------------------------------------------------------------------
# main() — exit codes map to verdicts
# ---------------------------------------------------------------------------

class TestMainExitCodes:
    def test_pass_verdict_gives_exit_0(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        mock_report = MagicMock()
        mock_report.global_verdict = "PASS"
        mock_cp = MagicMock()
        mock_cp.run.return_value = mock_report
        mock_cp.render.return_value = "rendered"
        monkeypatch.setattr("omniagis.cli.ColdPass", lambda: mock_cp)
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
        assert exc.value.code == 0

    def test_partial_pass_gives_exit_1(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        mock_report = MagicMock()
        mock_report.global_verdict = "PARTIAL PASS"
        mock_cp = MagicMock()
        mock_cp.run.return_value = mock_report
        mock_cp.render.return_value = "rendered"
        monkeypatch.setattr("omniagis.cli.ColdPass", lambda: mock_cp)
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
        assert exc.value.code == 1

    def test_no_pass_gives_exit_2(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        mock_report = MagicMock()
        mock_report.global_verdict = "NO PASS"
        mock_cp = MagicMock()
        mock_cp.run.return_value = mock_report
        mock_cp.render.return_value = "rendered"
        monkeypatch.setattr("omniagis.cli.ColdPass", lambda: mock_cp)
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
        assert exc.value.code == 2
