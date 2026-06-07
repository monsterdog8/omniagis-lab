# MODULE_006_ADJUDICATION

Manual adjudication scoring for metric classification in the OmniAGIS audit pipeline.

## Function

Processes completed adjudication CSV files where human reviewers have assigned ground-truth
labels to classifier predictions. Computes:

- **Confusion matrix**: `Dict[predicted_label, Counter[human_label]]`
- **Per-label precision/recall**: TP, predicted total, human total for each of 6 labels
- **Completion rate**: Fraction of rows with a non-empty `human_label`
- **Verification counts**: `value_verified`, `source_verified`, `replay_status` distributions
- **CSV I/O**: Read adjudication CSV → list of dicts; write confusion matrix → CSV

The 6 classification labels are:
`observed`, `computed`, `reported`, `simulated`, `symbolic`, `unsupported`

Extra labels tracked in confusion matrix output:
`invalid_parse`, `not_a_metric`

## Source

`gpts_core/adjudication.py` — copied verbatim to `exports/adjudication.py`

## Public API

| Symbol | Type | Description |
|---|---|---|
| `LABELS` | `List[str]` | 6 classification labels |
| `EXTRA_LABELS` | `List[str]` | Extra labels: `invalid_parse`, `not_a_metric` |
| `build_confusion_matrix(rows, pred_field, human_field)` | function | Build confusion matrix |
| `score_label(matrix, label)` | function | Per-label precision/recall |
| `score_adjudication(rows, labels, pred_field, human_field)` | function | Full adjudication scoring |
| `read_adjudication_csv(path)` | function | Read CSV → `List[Dict[str, str]]` |
| `write_confusion_matrix_csv(path, matrix, labels, extra)` | function | Write confusion matrix CSV |

### `score_adjudication` return structure

```python
{
    "generated_utc": str,
    "rows_total": int,
    "rows_completed": int,
    "completion_rate": float,          # in [0, 1]
    "per_label": {
        label: {
            "label": str,
            "sampled_predicted": int,
            "true_positive": int,
            "human_total": int,
            "precision": Optional[float],
            "recall": Optional[float],
        }
    },
    "value_verified_counts": Dict[str, int],
    "source_verified_counts": Dict[str, int],
    "replay_status_counts": Dict[str, int],
    "score": None,
    "production_status": "LOCKED",
    "verdict": "ADJUDICATION_SCORED_LOCAL_ONLY",
}
```

## Dependencies

- `csv` (stdlib)
- `pathlib` (stdlib)
- `collections` (stdlib)

## Coverage

82% (as measured against `tests/test_gpts_core.py::TestAdjudication`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — inter-rater agreement is computed locally; no external validation has been performed.

## Usage

```python
from modules.MODULE_006_ADJUDICATION.exports.adjudication import (
    score_adjudication, read_adjudication_csv, build_confusion_matrix
)
from pathlib import Path

rows = read_adjudication_csv(Path("adjudication_round1.csv"))
result = score_adjudication(rows)
print(result["completion_rate"])
print(result["per_label"]["observed"]["precision"])
```

See `examples/example.py` for runnable examples.
