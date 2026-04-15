# omniagis-lab
Fail-closed scientific validation framework for dynamical systems using return-time statistics and ε-robustness (OMNIÆGIS).

## Architecture

```
src/omniagis/
├── __init__.py
├── cli.py                     # CLI entry: python -m omniagis.cli <path>
├── core/
│   ├── __init__.py
│   ├── validator.py           # ε-robustness validator
│   ├── return_time.py         # return-time (Poincaré recurrence) statistics
│   └── classifier.py          # fail-closed verdict classifier
└── audit/
    ├── __init__.py
    ├── inventory.py           # file inventory & type classification
    ├── parsability.py         # Python parse/import checks
    ├── scorecard.py           # M1–M12 scorecard generator
    └── cold_pass.py           # Mode MAVERICK orchestrator → outputs A–G
tests/
├── test_validator.py
├── test_return_time.py
└── test_cold_pass.py
```

## Quick Start

```bash
pip install -e .
# or just install dependencies
pip install numpy pytest
```

## Mode MAVERICK Usage

```bash
# Audit the current directory
python -m omniagis.cli .
# (equivalent package entrypoint)
python -m omniagis .

# Audit a specific path with JSON output
python -m omniagis.cli /path/to/project --output json

# Exit codes: 0=PASS, 1=PARTIAL PASS, 2=NO PASS
```

## Benchmark Modes

```bash
# Single benchmark run (Pomeau-Manneville return-time experiment)
python -m omniagis benchmark --z 1.5 --epsilon 0.1 --output result.json

# Multi-ε benchmark sweep
python -m omniagis benchmark-sweep --z 1.5 --output sweep.json
```

## Output Sections (A–G)

| Section | Description |
|---------|-------------|
| **A**   | Structured inventory — table of all files with type, size, SHA-256 hash |
| **B**   | Scorecard M1–M12 — metrics table with PASS / PARTIAL PASS / NO PASS |
| **C**   | File-by-file audit — per-file parsability and import status |
| **D**   | Unresolved tensions — ghost imports, duplicates, syntax errors |
| **E**   | Cleanup plan — KEEP / REFACTOR / THROW / QUARANTINE matrix |
| **F**   | Validation plan — prioritised steps to reach global PASS |
| **G**   | Minimal core — essential files to keep |

## M1–M12 Scorecard

| ID  | Metric                          | Description |
|-----|---------------------------------|-------------|
| M1  | File inventory completeness     | Non-empty directory |
| M2  | Type separation                 | File types identified; < 50% UNKNOWN |
| M3  | Exact duplicates                | SHA-256 deduplication |
| M4  | Fake .py detection              | `.py` files with non-Python binary content |
| M5  | Pseudo-code                     | Files that parse but define no functions/classes |
| M6  | Ghost imports                   | Imports not resolvable in current environment |
| M7  | Missing dependencies            | requirements.txt packages vs. installed |
| M8  | Parsability                     | Fraction of `.py` files with valid syntax |
| M9  | Executability                   | Parse + no ghost imports |
| M10 | Classification coverage         | < 20% UNKNOWN files |
| M11 | KEEP/REFACTOR/THROW/QUARANTINE  | Disposition matrix |
| M12 | Global verdict (fail-closed)    | Any NO PASS → global NO PASS |

## OMNIÆGIS Core Components

### `EpsilonRobustnessValidator`
Validates a trajectory `[T, N]` against a reference using pointwise L2 distances.
Returns `PASS` if max distance ≤ ε, `PARTIAL PASS` if mean ≤ ε, else `NO PASS`.

```python
from omniagis.core.validator import EpsilonRobustnessValidator
import numpy as np

v = EpsilonRobustnessValidator(epsilon=1e-3)
result = v.validate(trajectory, reference)
print(result.status, result.max_dist)
```

### `ReturnTimeStatistics`
Computes Poincaré recurrence statistics for a 1-D scalar time series.

```python
from omniagis.core.return_time import ReturnTimeStatistics
import numpy as np

rts = ReturnTimeStatistics(tolerance=0.05)
indices = rts.find_returns(series, target_value=0.0)
stats = rts.compute_stats(indices)
verdict = rts.classify(stats, max_allowed_mean=50)
```

### `FailClosedClassifier`
Combines multiple verdicts using fail-closed logic: any `NO PASS` → `NO PASS`.

```python
from omniagis.core.classifier import FailClosedClassifier

clf = FailClosedClassifier()
verdict = clf.combine(["PASS", "PARTIAL PASS", "PASS"])  # → "PARTIAL PASS"
```
