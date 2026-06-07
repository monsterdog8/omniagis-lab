# MODULE_002_CLASSIFIER

Heuristic 6-label metric text classifier for the OmniAGIS audit pipeline.

## Function

Classifies extracted metric rows into audit categories using regex heuristics.
Labels require manual review before use as evidence.

The six labels are:
- **observed** — measurement or image/sensor data
- **computed** — derived via arithmetic or formula
- **reported** — configuration, threshold, status, or declared value
- **simulated** — synthetic, random, or mock data
- **symbolic** — mathematical constants or metaphoric/non-numeric context
- **unsupported** — strong claims, generic variables, or insufficient context

## Source

`gpts_core/classifier.py` — copied verbatim to `exports/classifier.py`

## Public API

| Symbol | Type | Description |
|---|---|---|
| `LABELS` | `List[str]` | Ordered list of 6 classification labels |
| `classify_metric` | function | Classify a metric row; returns `(label, reasons, confidence)` |

### `classify_metric` signature

```python
classify_metric(
    metric_name: str,
    value_raw: str,
    context: str = "",
    path: str = "",
    row_type: str = "metric",
    recalculable: Optional[str] = None,
    result: Optional[str] = None,
) -> Tuple[str, List[str], float]
```

Returns `(label, reasons, confidence)` where:
- `label` is one of the 6 strings in `LABELS`
- `reasons` is a list of reason-code strings explaining the classification
- `confidence` is a float in `[0, 1]`

## Dependencies

- `re` (stdlib)
- `math` (stdlib)

## Coverage

64% (as measured against `tests/test_gpts_core.py::TestClassifier`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — heuristic classifier; all labels require manual review before use as evidence.

## Usage

```python
from modules.MODULE_002_CLASSIFIER.exports.classifier import classify_metric, LABELS

label, reasons, conf = classify_metric("luminance_mean", "128.5", "measured from image histogram")
print(label, conf)
# observed 0.74
```

See `examples/example.py` for runnable examples.
