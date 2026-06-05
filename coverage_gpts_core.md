# coverage_gpts_core.md
## Measurement: coverage(gpts_core)
**Command:** `pytest tests/test_gpts_core.py --cov=gpts_core`
**Date:** 2026-06-05
**Tests run:** 114 passed, 0 failed
**Claim ceiling:** LOCAL_MEASURED — not proof of correctness

| MODULE | LINES | COVERED | MISSED | PERCENT |
|---|---|---|---|---|
| `__init__.py` | 14 | 14 | 0 | 100% |
| `adjudication.py` | 44 | 38 | 6 | 82% |
| `audit_claims.py` | 351 | 215 | 136 | 52% |
| `benchmark.py` | 141 | 101 | 40 | 70% |
| `classifier.py` | 59 | 40 | 19 | 64% |
| `cli.py` | 183 | 0 | 183 | 0% |
| `coherence.py` | 189 | 155 | 34 | 78% |
| `dynamics.py` | 110 | 93 | 17 | 82% |
| `evidence.py` | 170 | 124 | 46 | 66% |
| `gate.py` | 117 | 109 | 8 | 91% |
| `ledger.py` | 101 | 71 | 30 | 64% |
| `manifest.py` | 64 | 59 | 5 | 90% |
| `promotion.py` | 122 | 86 | 36 | 65% |
| `score_report.py` | 280 | 210 | 70 | 70% |
| `signals.py` | 171 | 128 | 43 | 69% |
| `spectral_gap.py` | 126 | 106 | 20 | 81% |
| **TOTAL** | **2242** | **1549** | **693** | **65%** |

## Notes
- `cli.py` = 0%: CLI module requires subprocess/argument parsing; not tested here
- `audit_claims.py` = 52%: complex branching in report generation; many paths not exercised
- All other modules ≥ 52%
- Measurement is reproducible: run `pytest tests/test_gpts_core.py --cov=gpts_core`
