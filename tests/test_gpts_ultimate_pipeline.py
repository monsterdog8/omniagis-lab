"""Tests for gpts_ultimate.pipeline — GptsUltimatePipeline end-to-end."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gpts_ultimate.pipeline import GptsUltimatePipeline, _load_trajectory

SINE = np.sin(np.linspace(0, 8 * np.pi, 512))
SEEDS = list(range(5, 12))


# ---------------------------------------------------------------------------
# _load_trajectory
# ---------------------------------------------------------------------------
class TestLoadTrajectory:
    def test_array_passthrough(self):
        arr = _load_trajectory(SINE)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(SINE)

    def test_list_accepted(self):
        arr = _load_trajectory(SINE.tolist())
        assert isinstance(arr, np.ndarray)

    def test_csv_load(self, tmp_path):
        p = tmp_path / "traj.csv"
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for v in SINE[:50]:
                writer.writerow([v])
        arr = _load_trajectory(p)
        assert len(arr) == 50

    def test_csv_wrong_column_raises(self, tmp_path):
        p = tmp_path / "traj.csv"
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x"])
            writer.writerow([1.0])
        with pytest.raises(ValueError):
            _load_trajectory(p, column="value")


# ---------------------------------------------------------------------------
# GptsUltimatePipeline
# ---------------------------------------------------------------------------
class TestGptsUltimatePipeline:
    def test_run_returns_report(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="test_run", seeds=SEEDS, n_null=5)
        assert "run_id" in report
        assert "signal" in report
        assert "evidence" in report
        assert "tribunal" in report

    def test_report_run_id(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="my_run", seeds=SEEDS, n_null=5)
        assert report["run_id"] == "my_run"

    def test_task_id_defaults_to_run_id(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="autorun", seeds=SEEDS, n_null=5)
        assert report["task_id"] == "autorun"

    def test_task_id_custom(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", task_id="my_task", seeds=SEEDS, n_null=5)
        assert report["task_id"] == "my_task"

    def test_claim_ceiling(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        assert report["claim_ceiling"] == "LOCAL_SIMULATION_ONLY"

    def test_production_unlocked_false(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        assert report["production_unlocked"] is False

    def test_ledger_entry_logged(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        assert "ledger_entry" in report
        assert "entry_hash" in report["ledger_entry"]

    def test_ledger_accumulates(self):
        pipe = GptsUltimatePipeline()
        pipe.run(SINE, run_id="run_a", seeds=SEEDS, n_null=5)
        pipe.run(SINE, run_id="run_b", seeds=SEEDS, n_null=5)
        assert len(pipe.ledger.entries()) == 2

    def test_persist_ledger(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        pipe = GptsUltimatePipeline(ledger_path=path)
        pipe.run(SINE, run_id="persist", seeds=SEEDS, n_null=5)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_from_csv(self, tmp_path):
        p = tmp_path / "signal.csv"
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for v in SINE:
                writer.writerow([v])
        pipe = GptsUltimatePipeline()
        report = pipe.run(p, run_id="csv_run", seeds=SEEDS, n_null=5)
        assert report["n_samples"] == len(SINE)

    def test_signal_has_structure_score(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        assert "structure_score" in report["signal"]

    def test_evidence_dict_keys(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        ev = report["evidence"]
        assert "final_score" in ev
        assert "public_claim_allowed" in ev

    def test_tribunal_final_verdict(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        valid = {"REFUTED_VS_NULL", "OPEN_PROBLEM", "EMPIRICAL_SIGNAL", "PASS_LOCAL"}
        assert report["tribunal"]["final_verdict"] in valid

    def test_n_samples_matches(self):
        pipe = GptsUltimatePipeline()
        report = pipe.run(SINE, run_id="r1", seeds=SEEDS, n_null=5)
        assert report["n_samples"] == len(SINE)
