"""Manual adjudication scoring.

Reads a CSV with classifier predictions and human labels, computes
confusion matrix, precision/recall per label, and overall completion rate.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


LABELS = ["observed", "computed", "reported", "simulated", "symbolic", "unsupported"]
EXTRA_LABELS = ["invalid_parse", "not_a_metric"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_confusion_matrix(
    rows: List[Dict[str, str]],
    pred_field: str = "label",
    human_field: str = "human_label",
) -> Dict[str, Counter]:
    """Build confusion matrix as dict[predicted_label] -> Counter[human_label]."""
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        pred = r.get(pred_field, "").strip()
        human = r.get(human_field, "").strip()
        if pred and human:
            matrix[pred][human] += 1
    return dict(matrix)


def score_label(
    matrix: Dict[str, Counter],
    label: str,
) -> Dict[str, Any]:
    """Compute precision/recall/TP for a single label."""
    predicted_total = sum(matrix.get(label, Counter()).values())
    tp = matrix.get(label, Counter())[label]
    human_total = sum(matrix.get(p, Counter())[label] for p in matrix)
    precision = tp / predicted_total if predicted_total else None
    recall = tp / human_total if human_total else None
    return {
        "label": label,
        "sampled_predicted": predicted_total,
        "true_positive": tp,
        "human_total": human_total,
        "precision": precision,
        "recall": recall,
    }


def score_adjudication(
    rows: List[Dict[str, str]],
    labels: Sequence[str] = LABELS,
    pred_field: str = "label",
    human_field: str = "human_label",
) -> Dict[str, Any]:
    """Score a completed adjudication CSV."""
    total = len(rows)
    completed = [r for r in rows if r.get(human_field, "").strip()]
    matrix = build_confusion_matrix(rows, pred_field, human_field)
    per_label = {lab: score_label(matrix, lab) for lab in labels}
    value_verified = Counter(r.get("value_verified", "").strip() for r in completed)
    source_verified = Counter(r.get("source_verified", "").strip() for r in completed)
    replay_status = Counter(r.get("replay_status", "").strip() for r in completed)
    return {
        "generated_utc": _utc_now(),
        "rows_total": total,
        "rows_completed": len(completed),
        "completion_rate": len(completed) / total if total else 0.0,
        "per_label": per_label,
        "value_verified_counts": dict(value_verified),
        "source_verified_counts": dict(source_verified),
        "replay_status_counts": dict(replay_status),
        "score": None,
        "production_status": "LOCKED",
        "verdict": "ADJUDICATION_SCORED_LOCAL_ONLY",
    }


def read_adjudication_csv(path: Path) -> List[Dict[str, str]]:
    """Read adjudication CSV file into list of dicts."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_confusion_matrix_csv(
    path: Path,
    matrix: Dict[str, Counter],
    labels: Sequence[str] = LABELS,
    extra: Sequence[str] = EXTRA_LABELS,
) -> None:
    """Write confusion matrix as CSV."""
    all_labels = list(labels) + list(extra)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier_label"] + all_labels)
        for lab in labels:
            w.writerow([lab] + [matrix.get(lab, Counter())[h] for h in all_labels])
