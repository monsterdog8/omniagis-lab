# EXOCHRONOS Module Registry

**Version:** 1.0.0  
**Date:** 2026-06-07  
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF  
**Extraction mode:** EXTRACTION_ONLY — no new architecture

## Summary

18 modules extracted and packaged from the `gpts_core` Python package and governance artifacts.

| Category | Count | Coverage |
|---|---|---|
| Code modules (gpts_core) | 16 | 65% avg |
| Governance modules | 2 | N/A |
| **Total** | **18** | — |

## Module Index

| ID | Name | Source | Coverage | Stdlib only |
|---|---|---|---|---|
| MODULE_001 | SIGNALS | signals.py | 69% | No (numpy/scipy) |
| MODULE_002 | CLASSIFIER | classifier.py | 64% | Yes |
| MODULE_003 | EVIDENCE | evidence.py | 66% | Yes |
| MODULE_004 | GATE | gate.py | 91% | Yes |
| MODULE_005 | LEDGER | ledger.py | 64% | Yes |
| MODULE_006 | ADJUDICATION | adjudication.py | 82% | Yes |
| MODULE_007 | MANIFEST | manifest.py | 90% | Yes |
| MODULE_008 | BENCHMARK | benchmark.py | 70% | Yes |
| MODULE_009 | PROMOTION | promotion.py | 65% | Yes |
| MODULE_010 | COHERENCE | coherence.py | 78% | Yes |
| MODULE_011 | AUDIT_CLAIMS | audit_claims.py | 52% | Yes |
| MODULE_012 | SCORE_REPORT | score_report.py | 70% | Yes |
| MODULE_013 | SPECTRAL_GAP | spectral_gap.py | 81% | No (numpy/scipy) |
| MODULE_014 | DYNAMICS | dynamics.py | 82% | Yes |
| MODULE_015 | CLI | cli.py | 0%* | No |
| MODULE_016 | CORE_INIT | __init__.py | 100% | No |
| MODULE_017 | EXOCHRONOS_STATE | EXOCHRONOS_STATE_v1.json | N/A | — |
| MODULE_018 | CLASSIFICATION_MATRIX | classification_matrix_schema_v1.json | N/A | — |

*CLI at 0%: subprocess-only interface, not unit-testable without refactor

## Each module contains

```
MODULE_XXX_NAME/
  manifest.json     ← name, version, API, dependencies, status
  README.md         ← documentation
  schema.json       ← JSON Schema for inputs/outputs
  tests/            ← extracted from tests/test_gpts_core.py
  examples/         ← runnable usage examples
  exports/          ← verbatim source file copy
```

## Replay

```bash
python -m pytest tests/test_gpts_core.py --cov=gpts_core  # 114 tests, 65%
python -m pytest tests/test_equivalence_*.py               # 21 tests, all pass
```

## Claim ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All measurements are locally reproducible. No external validation has been performed.
