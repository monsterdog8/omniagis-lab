"""Minimal usage examples for MODULE_006_ADJUDICATION.

Run from repository root:
    python modules/MODULE_006_ADJUDICATION/examples/example.py
"""
from __future__ import annotations

import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

from adjudication import (
    LABELS, EXTRA_LABELS,
    build_confusion_matrix, score_label, score_adjudication,
    read_adjudication_csv, write_confusion_matrix_csv,
)


# ---------------------------------------------------------------------------
# Example 1: Score a batch of adjudicated rows
# ---------------------------------------------------------------------------

def example_score_adjudication():
    print("=== Example 1: score_adjudication() on a sample batch ===")

    rows = [
        {"label": "observed",    "human_label": "observed",    "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "observed",    "human_label": "observed",    "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "computed",    "human_label": "computed",    "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "computed",    "human_label": "observed",    "value_verified": "no",  "source_verified": "no",  "replay_status": "FAIL"},
        {"label": "reported",    "human_label": "reported",    "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "simulated",   "human_label": "simulated",   "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "symbolic",    "human_label": "symbolic",    "value_verified": "no",  "source_verified": "yes", "replay_status": "PASS"},
        {"label": "unsupported", "human_label": "unsupported", "value_verified": "yes", "source_verified": "yes", "replay_status": "PASS"},
        {"label": "unsupported", "human_label": "computed",    "value_verified": "no",  "source_verified": "no",  "replay_status": "FAIL"},
        {"label": "simulated",   "human_label": "",            "value_verified": "",    "source_verified": "",    "replay_status": ""},
    ]

    result = score_adjudication(rows)

    print(f"  rows_total        : {result['rows_total']}")
    print(f"  rows_completed    : {result['rows_completed']}")
    print(f"  completion_rate   : {result['completion_rate']:.2%}")
    print(f"  verdict           : {result['verdict']}")
    print(f"  production_status : {result['production_status']}")
    print()
    print(f"  {'label':<14} {'precision':<12} {'recall':<12} {'TP'}")
    print("  " + "-" * 48)
    for lab in LABELS:
        s = result["per_label"][lab]
        prec = f"{s['precision']:.3f}" if s["precision"] is not None else "N/A"
        rec  = f"{s['recall']:.3f}"    if s["recall"]    is not None else "N/A"
        print(f"  {lab:<14} {prec:<12} {rec:<12} {s['true_positive']}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Confusion matrix and per-label score_label()
# ---------------------------------------------------------------------------

def example_confusion_matrix():
    print("=== Example 2: build_confusion_matrix() and score_label() ===")

    rows = [
        {"label": "observed",  "human_label": "observed"},
        {"label": "observed",  "human_label": "observed"},
        {"label": "observed",  "human_label": "computed"},   # misclassification
        {"label": "computed",  "human_label": "computed"},
        {"label": "simulated", "human_label": "simulated"},
    ]

    matrix = build_confusion_matrix(rows)

    print("  Confusion matrix (predicted -> human counts):")
    for pred_lab, counter in sorted(matrix.items()):
        for human_lab, count in sorted(counter.items()):
            print(f"    pred={pred_lab:<12} human={human_lab:<12} count={count}")

    print()
    s = score_label(matrix, "observed")
    print(f"  score_label('observed'):")
    print(f"    sampled_predicted : {s['sampled_predicted']}")
    print(f"    true_positive     : {s['true_positive']}")
    print(f"    precision         : {s['precision']:.4f}" if s["precision"] is not None else "    precision         : N/A")
    print(f"    recall            : {s['recall']:.4f}"    if s["recall"]    is not None else "    recall            : N/A")
    print()


# ---------------------------------------------------------------------------
# Example 3: CSV round-trip (read, score, write confusion matrix)
# ---------------------------------------------------------------------------

def example_csv_roundtrip():
    print("=== Example 3: CSV read/write round-trip ===")

    csv_content = (
        "label,human_label,value_verified,source_verified,replay_status\n"
        "observed,observed,yes,yes,PASS\n"
        "computed,computed,yes,yes,PASS\n"
        "reported,reported,yes,yes,PASS\n"
        "simulated,simulated,yes,yes,PASS\n"
        "symbolic,symbolic,no,yes,PASS\n"
        "unsupported,computed,no,no,FAIL\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # Write source CSV
        src = tmp / "adjudication.csv"
        src.write_text(csv_content)

        # Read and score
        rows = read_adjudication_csv(src)
        result = score_adjudication(rows)
        print(f"  read {len(rows)} rows, completion_rate={result['completion_rate']:.2%}")

        # Write confusion matrix CSV
        matrix = build_confusion_matrix(rows)
        out = tmp / "confusion_matrix.csv"
        write_confusion_matrix_csv(out, matrix)
        header_line = out.read_text().splitlines()[0]
        print(f"  confusion matrix CSV header: {header_line}")
        print(f"  extra labels tracked: {EXTRA_LABELS}")
    print()


if __name__ == "__main__":
    example_score_adjudication()
    example_confusion_matrix()
    example_csv_roundtrip()
