# MODULE_008_BENCHMARK

## Purpose

Benchmark utilities: stdlib-only linear baseline model, MSE stability scoring, and prediction lock comparison.

Provides a complete linear regression pipeline (gradient descent, no external deps), a weighted MSE-based scoring function with missing/invalid/range penalties, and a structured comparator for pre-registered numeric prediction locks used in shadow observation gates.

## Source

`gpts_core/benchmark.py`

## Public API

| Function | Signature | Returns | Description |
|---|---|---|---|
| `minmax_fit` | `(matrix: List[List[float]]) -> Tuple[List[float], List[float]]` | `Tuple[List[float], List[float]]` | Fit per-column min/max scaler |
| `minmax_transform` | `(matrix, mins, maxs) -> List[List[float]]` | `List[List[float]]` | Apply min-max scaling |
| `train_linear` | `(X, y, epochs=5000, lr=0.05) -> Tuple[List[float], float]` | `Tuple[List[float], float]` | Gradient descent linear regression |
| `predict_linear` | `(X, weights, bias, clip=True) -> List[float]` | `List[float]` | Linear model prediction, optional [0,1] clip |
| `read_feature_csv` | `(path: Path, feature_cols: List[str]) -> List[List[float]]` | `List[List[float]]` | Read feature matrix from CSV |
| `mse_score` | `(predictions, truth, missing_penalty, invalid_penalty, range_penalty) -> float` | `float` | Weighted stability score in [0,1] |
| `read_score_csv` | `(path: Path) -> Dict[str, float]` | `Dict[str, float]` | Read sample_id → stability_score from CSV |
| `compare_prediction_lock` | `(prediction_lock: Dict, observation: Dict) -> Dict` | `Dict[str, Any]` | Shadow gate G2 prediction lock comparator |

## Dependencies

All stdlib: `csv`, `pathlib`, `math`, `hashlib`, `json`, `datetime`

## Coverage

70%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All outputs are local computations. No external validation is claimed. Production is never automatically unlocked by the prediction lock comparator.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_008.py` — extracted from `tests/test_gpts_core.py::TestBenchmark`
