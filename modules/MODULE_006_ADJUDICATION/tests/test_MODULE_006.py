"""Tests for MODULE_006_ADJUDICATION.

Extracted from tests/test_gpts_core.py::TestAdjudication.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestAdjudication:
    def _rows(self):
        return [
            {"label": "simulated", "human_label": "simulated"},
            {"label": "observed", "human_label": "observed"},
            {"label": "computed", "human_label": "simulated"},  # wrong
            {"label": "reported", "human_label": "reported"},
        ]

    def test_build_confusion_matrix(self):
        from adjudication import build_confusion_matrix
        matrix = build_confusion_matrix(self._rows())
        assert isinstance(matrix, dict)
        assert "simulated" in matrix

    def test_score_adjudication_keys(self):
        from adjudication import score_adjudication
        result = score_adjudication(self._rows())
        assert isinstance(result, dict)
        assert "per_label" in result or "precision_macro" in result or "macro_precision" in result or "overall" in result

    def test_score_adjudication_empty(self):
        from adjudication import score_adjudication
        result = score_adjudication([])
        assert isinstance(result, dict)

    def test_perfect_adjudication(self):
        from adjudication import score_adjudication
        rows = [{"label": "observed", "human_label": "observed"}] * 10
        result = score_adjudication(rows)
        assert isinstance(result, dict)

    def test_read_adjudication_csv(self, tmp_path):
        from adjudication import read_adjudication_csv
        f = tmp_path / "adj.csv"
        f.write_text("label,human_label\nsimulated,simulated\nobserved,observed\n")
        rows = read_adjudication_csv(f)
        assert len(rows) == 2
        assert rows[0]["label"] == "simulated"

    # Additional coverage tests

    def test_labels_constant(self):
        from adjudication import LABELS
        assert set(LABELS) == {"observed", "computed", "reported", "simulated", "symbolic", "unsupported"}
        assert len(LABELS) == 6

    def test_extra_labels_constant(self):
        from adjudication import EXTRA_LABELS
        assert "invalid_parse" in EXTRA_LABELS
        assert "not_a_metric" in EXTRA_LABELS

    def test_build_confusion_matrix_correct_counts(self):
        from adjudication import build_confusion_matrix
        rows = [
            {"label": "observed", "human_label": "observed"},
            {"label": "observed", "human_label": "observed"},
            {"label": "observed", "human_label": "computed"},
        ]
        matrix = build_confusion_matrix(rows)
        assert matrix["observed"]["observed"] == 2
        assert matrix["observed"]["computed"] == 1

    def test_build_confusion_matrix_skips_empty(self):
        from adjudication import build_confusion_matrix
        rows = [
            {"label": "observed", "human_label": ""},
            {"label": "", "human_label": "observed"},
            {"label": "simulated", "human_label": "simulated"},
        ]
        matrix = build_confusion_matrix(rows)
        # Only the last row with both pred and human non-empty should be counted
        assert "simulated" in matrix
        assert matrix["simulated"]["simulated"] == 1
        assert "observed" not in matrix or matrix.get("observed", {}).get("", 0) == 0

    def test_score_label_precision_recall(self):
        from adjudication import build_confusion_matrix, score_label
        rows = [
            {"label": "observed", "human_label": "observed"},
            {"label": "observed", "human_label": "observed"},
            {"label": "observed", "human_label": "simulated"},
            {"label": "simulated", "human_label": "observed"},
        ]
        matrix = build_confusion_matrix(rows)
        s = score_label(matrix, "observed")
        # precision = 2/3, recall = 2/3
        assert s["sampled_predicted"] == 3
        assert s["true_positive"] == 2
        assert abs(s["precision"] - 2 / 3) < 1e-9
        assert abs(s["recall"] - 2 / 3) < 1e-9

    def test_score_label_no_predictions(self):
        from adjudication import build_confusion_matrix, score_label
        rows = [{"label": "simulated", "human_label": "simulated"}]
        matrix = build_confusion_matrix(rows)
        s = score_label(matrix, "observed")
        assert s["precision"] is None
        assert s["recall"] is None
        assert s["sampled_predicted"] == 0

    def test_score_adjudication_completion_rate(self):
        from adjudication import score_adjudication
        rows = [
            {"label": "observed", "human_label": "observed"},
            {"label": "computed", "human_label": ""},
            {"label": "reported", "human_label": "reported"},
        ]
        result = score_adjudication(rows)
        assert result["rows_total"] == 3
        assert result["rows_completed"] == 2
        assert abs(result["completion_rate"] - 2 / 3) < 1e-9

    def test_score_adjudication_production_locked(self):
        from adjudication import score_adjudication
        result = score_adjudication(self._rows())
        assert result.get("production_status") == "LOCKED"
        assert result.get("verdict") == "ADJUDICATION_SCORED_LOCAL_ONLY"

    def test_score_adjudication_per_label_all_labels(self):
        from adjudication import score_adjudication, LABELS
        result = score_adjudication(self._rows())
        if "per_label" in result:
            for lab in LABELS:
                assert lab in result["per_label"]

    def test_write_confusion_matrix_csv(self, tmp_path):
        from adjudication import build_confusion_matrix, write_confusion_matrix_csv
        rows = [
            {"label": "observed", "human_label": "observed"},
            {"label": "simulated", "human_label": "simulated"},
            {"label": "computed", "human_label": "observed"},
        ]
        matrix = build_confusion_matrix(rows)
        out = tmp_path / "confusion.csv"
        write_confusion_matrix_csv(out, matrix)
        assert out.exists()
        content = out.read_text()
        assert "classifier_label" in content
        assert "observed" in content
        assert "simulated" in content

    def test_write_confusion_matrix_csv_header(self, tmp_path):
        from adjudication import build_confusion_matrix, write_confusion_matrix_csv, LABELS, EXTRA_LABELS
        rows = [{"label": "reported", "human_label": "reported"}]
        matrix = build_confusion_matrix(rows)
        out = tmp_path / "cm.csv"
        write_confusion_matrix_csv(out, matrix)
        lines = out.read_text().splitlines()
        header = lines[0].split(",")
        assert header[0] == "classifier_label"
        for lab in LABELS:
            assert lab in header
        for lab in EXTRA_LABELS:
            assert lab in header

    def test_read_adjudication_csv_all_fields(self, tmp_path):
        from adjudication import read_adjudication_csv
        f = tmp_path / "full.csv"
        f.write_text(
            "label,human_label,value_verified,source_verified,replay_status\n"
            "observed,observed,yes,yes,PASS\n"
            "simulated,simulated,no,yes,FAIL\n"
        )
        rows = read_adjudication_csv(f)
        assert len(rows) == 2
        assert rows[0]["value_verified"] == "yes"
        assert rows[1]["replay_status"] == "FAIL"

    def test_score_adjudication_verified_counts(self, tmp_path):
        from adjudication import score_adjudication
        rows = [
            {"label": "observed", "human_label": "observed",
             "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
            {"label": "computed", "human_label": "computed",
             "value_verified": "no", "source_verified": "yes", "replay_status": "FAIL"},
        ]
        result = score_adjudication(rows)
        assert result["value_verified_counts"].get("yes", 0) == 1
        assert result["replay_status_counts"].get("PASS", 0) == 1
