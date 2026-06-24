"""End-to-end analysis pipeline — gpts_ultimate.

GptsUltimatePipeline sequences:
  1. signals.analyze()              — 17-key signal fingerprint
  2. gate.compute_evidence_score()  — evidence scoring from signal quality
  3. tribunal.run_full_tribunal()   — agnostic + optional BERZERKER verdict
  4. AuditLedger.log()              — tamper-evident chain entry
  5. return consolidated report dict

Claim ceiling: LOCAL_SIMULATION_ONLY
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from gpts_ultimate.signals import analyze
from gpts_ultimate.gate import compute_evidence_score, from_gate_counts
from gpts_ultimate.tribunal import run_full_tribunal
from gpts_ultimate.ledger import AuditLedger


def _load_trajectory(source: Union[str, Path, np.ndarray, List[float]], column: str = "value") -> np.ndarray:
    """Accept a 1-D array, list, or path to a CSV file."""
    if isinstance(source, np.ndarray):
        return source.astype(float)
    if isinstance(source, list):
        return np.array(source, dtype=float)
    path = Path(source)
    values: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, KeyError):
                pass
    if not values:
        raise ValueError(f"No numeric values found in column {column!r} of {path}")
    return np.array(values, dtype=float)


class GptsUltimatePipeline:
    """End-to-end signal analysis pipeline with ledger integration."""

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        chained: bool = True,
        sr: float = 256.0,
    ) -> None:
        self._ledger = AuditLedger(path=ledger_path, chained=chained)
        self._sr = sr

    def run(
        self,
        trajectory: Union[str, Path, np.ndarray, List[float]],
        run_id: str,
        task_id: str = "auto",
        lorenz_ensemble: Optional[np.ndarray] = None,
        seeds: Sequence[int] = range(5, 15),
        n_null: int = 20,
        csv_column: str = "value",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline and return a consolidated report.

        Parameters
        ----------
        trajectory:
            1-D signal as ndarray, list of floats, or path to CSV.
        run_id:
            Unique identifier for this run (logged to ledger).
        task_id:
            Human-readable task label (defaults to run_id).
        lorenz_ensemble:
            Optional (T, N) ensemble; if provided, BERZERKER tribunal is also run.
        seeds:
            Seeds for bootstrap / null sampling.
        n_null:
            Number of null surrogates for agnostic tribunal.
        csv_column:
            Column name when trajectory is a CSV path (default: "value").
        output_path:
            If set, BERZERKER report is also written to this JSON file.
        """
        if task_id == "auto":
            task_id = run_id

        arr = _load_trajectory(trajectory, column=csv_column)

        # Step 1: Signal fingerprint
        sig = analyze(arr.tolist(), sr=self._sr)

        # Step 2: Evidence scoring from signal quality proxy
        n = len(arr)
        inp = from_gate_counts(
            expected=n,
            present=n,
            valid=n,
            safety=min(1.0, sig.get("structure_score", 0.5) + 0.3),
        )
        evidence = compute_evidence_score(inp)

        # Step 3: Tribunal
        tribunal = run_full_tribunal(
            trajectory=arr,
            lorenz_ensemble=lorenz_ensemble,
            seeds=list(seeds),
            n_null=n_null,
            output_path=output_path,
        )

        # Step 4: Ledger entry
        entry = self._ledger.log(
            event="PIPELINE_RUN",
            data={
                "run_id": run_id,
                "task_id": task_id,
                "n_samples": n,
                "structure_score": sig.get("structure_score"),
                "evidence_final_score": evidence.final_score,
                "final_verdict": tribunal["final_verdict"],
                "claim_ceiling": tribunal["claim_ceiling"],
            },
            audit=True,
        )

        return {
            "run_id": run_id,
            "task_id": task_id,
            "n_samples": n,
            "signal": sig,
            "evidence": evidence.to_dict(),
            "tribunal": tribunal,
            "ledger_entry": entry,
            "claim_ceiling": "LOCAL_SIMULATION_ONLY",
            "production_unlocked": False,
            "timestamp": time.time(),
        }

    @property
    def ledger(self) -> AuditLedger:
        return self._ledger
