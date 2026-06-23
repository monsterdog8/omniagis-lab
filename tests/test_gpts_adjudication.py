"""Tests for gpts_core.adjudication — confusion matrix and scoring."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from gpts_core.adjudication import (
    build_confusion_matrix,
    score_label,
    score_adjudication,
    read_adjudication_csv,
    write_confusion_matrix_csv,
    LABELS,
)


# ---------------------------------------------------------------------------
# build_confusion_matrix
# ---------------------------------------------------------------------------

class TestBuildConfusionMatrix:
    def test_basic_matrix(self):
        rows = [
            {"label": "computed", "human_label": "computed"},
            {"label": "computed", "human_label": "reported"},
            {"label": "observed", "human_label": "observed"},
        ]
        matrix = build_confusion_matrix(rows)
        assert matrix["computed"]["computed"] == 1
        assert matrix["computed"]["reported"] == 1
        assert matrix["observed"]["observed"] == 1

    def test_empty_rows(self):
        matrix = build_confusion_matrix([])
        assert matrix == {}

    def test_missing_fields_skipped(self):
        rows = [
            {"label": "computed"},
            {"human_label": "observed"},
            {"label": "", "human_label": "computed"},
        ]
        matrix = build_confusion_matrix(rows)
        assert matrix == {}

    def test_custom_field_names(self):
        rows = [{"pred": "computed", "human": "computed"}]
        matrix = build_confusion_matrix(rows, pred_field="pred", human_field="human")
        assert matrix["computed"]["computed"] == 1


# ---------------------------------------------------------------------------
# score_label
# ---------------------------------------------------------------------------

class TestScoreLabel:
    def test_perfect_precision_recall(self):
        matrix = {"computed": Counter({"computed": 3})}
        result = score_label(matrix, "computed")
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["true_positive"] == 3

    def test_zero_predictions_returns_none_precision(self):
        matrix = {}
        result = score_label(matrix, "computed")
        assert result["precision"] is None
        assert result["recall"] is None
        assert result["true_positive"] == 0

    def test_partial_precision(self):
        matrix = {"computed": Counter({"computed": 2, "reported": 1})}
        result = score_label(matrix, "computed")
        assert result["precision"] == pytest.approx(2 / 3)

    def test_partial_recall(self):
        matrix = {
            "computed": Counter({"computed": 2}),
            "reported": Counter({"computed": 1}),
        }
        result = score_label(matrix, "computed")
        assert result["recall"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# score_adjudication
# ---------------------------------------------------------------------------

class TestScoreAdjudication:
    def _make_rows(self, n_complete=3, n_incomplete=1):
        rows = []
        for i in range(n_complete):
            rows.append({
                "label": "computed",
                "human_label": "computed",
                "value_verified": "YES",
                "source_verified": "YES",
                "replay_status": "PASS",
            })
        for i in range(n_incomplete):
            rows.append({
                "label": "observed",
                "human_label": "",
            })
        return rows

    def test_completion_rate(self):
        rows = self._make_rows(n_complete=3, n_incomplete=1)
        result = score_adjudication(rows)
        assert result["rows_total"] == 4
        assert result["rows_completed"] == 3
        assert result["completion_rate"] == pytest.approx(0.75)

    def test_empty_rows(self):
        result = score_adjudication([])
        assert result["rows_total"] == 0
        assert result["completion_rate"] == 0.0

    def test_per_label_has_all_labels(self):
        rows = self._make_rows()
        result = score_adjudication(rows)
        for label in LABELS:
            assert label in result["per_label"]

    def test_production_status_locked(self):
        result = score_adjudication([])
        assert result["production_status"] == "LOCKED"
        assert result["verdict"] == "ADJUDICATION_SCORED_LOCAL_ONLY"
        assert result["score"] is None

    def test_value_verified_counts(self):
        rows = self._make_rows(n_complete=3, n_incomplete=0)
        result = score_adjudication(rows)
        assert result["value_verified_counts"].get("YES", 0) == 3

    def test_replay_status_counts(self):
        rows = self._make_rows(n_complete=2, n_incomplete=0)
        result = score_adjudication(rows)
        assert result["replay_status_counts"].get("PASS", 0) == 2

    def test_generated_utc_present(self):
        result = score_adjudication([])
        assert "generated_utc" in result
        assert "T" in result["generated_utc"]


# ---------------------------------------------------------------------------
# read_adjudication_csv / write_confusion_matrix_csv
# ---------------------------------------------------------------------------

class TestCsvIO:
    def test_read_adjudication_csv(self, tmp_path):
        p = tmp_path / "adj.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "human_label"])
            w.writeheader()
            w.writerow({"label": "computed", "human_label": "computed"})
        rows = read_adjudication_csv(p)
        assert len(rows) == 1
        assert rows[0]["label"] == "computed"

    def test_write_confusion_matrix_csv(self, tmp_path):
        p = tmp_path / "cm.csv"
        matrix = {
            "computed": Counter({"computed": 5, "reported": 1}),
            "observed": Counter({"observed": 3}),
        }
        write_confusion_matrix_csv(p, matrix)
        assert p.exists()
        rows = list(csv.reader(p.open(encoding="utf-8")))
        assert rows[0][0] == "classifier_label"
