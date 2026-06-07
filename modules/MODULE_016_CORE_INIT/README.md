# MODULE_016_CORE_INIT

**Source:** `gpts_core/__init__.py`
**Version:** 1.0.0
**Coverage:** 100%
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF

## Description

Package entry point for `gpts_core`. Re-exports all public symbols from all 15 sub-modules so that callers can import directly from `gpts_core` without knowing which sub-module hosts each function.

`spectral_gap` symbols are intentionally omitted from `__all__` (noted as lazy) because that module requires `numpy`. Import them directly from `gpts_core.spectral_gap` when needed.

## Exported Symbol Groups

| Group | Module | Key Symbols |
|-------|--------|-------------|
| signals | `gpts_core.signals` | `analyze_signal`, `structure_score` |
| classifier | `gpts_core.classifier` | `classify_metric`, `LABELS` |
| evidence | `gpts_core.evidence` | `build_raw_record`, `validate_raw_records`, `sha256_text`, `sha256_file` |
| gate | `gpts_core.gate` | `classify_claim`, `compute_evidence_score`, `from_gate_counts`, `maturity_map`, `proof_firewall` |
| ledger | `gpts_core.ledger` | `AuditLedger`, `ContextCitationLock` |
| adjudication | `gpts_core.adjudication` | `score_adjudication`, `build_confusion_matrix` |
| manifest | `gpts_core.manifest` | `build_manifest`, `write_manifest`, `hash_file`, `kind_for` |
| benchmark | `gpts_core.benchmark` | `train_linear`, `predict_linear`, `mse_score`, `compare_prediction_lock` |
| promotion | `gpts_core.promotion` | `evaluate_promotion`, `replay_jsonl_ledger`, `validate_seal_ledger` |
| coherence | `gpts_core.coherence` | `shannon_entropy`, `mutual_information`, `global_coherence`, `build_coherence_passport` |
| audit_claims | `gpts_core.audit_claims` | `audit_claim`, `inspect_document`, `batch_audit`, `AuditReport` |
| score_report | `gpts_core.score_report` | `evaluate_report`, `score_audit_report` |
| dynamics | `gpts_core.dynamics` | `ContinuumState`, `step_continuum`, `continuum_energy`, `run_continuum`, `FractalState`, `FractalEngine` |

## Usage

```python
# Import any public symbol directly from gpts_core
from gpts_core import classify_claim, compute_evidence_score, ContinuumState

# Classify a claim
result = classify_claim("This is a lab-only prototype")
print(result.status)

# Use dynamics
state = ContinuumState()
from gpts_core import step_continuum
next_state = step_continuum(state, noise_std=0.0)

# spectral_gap requires numpy — import directly from sub-module
from gpts_core.spectral_gap import run_pipeline
```

## Dependencies

All 15 other gpts_core sub-modules. `spectral_gap` is imported lazily (requires `numpy`).

## Files

```
MODULE_016_CORE_INIT/
  manifest.json
  README.md
  schema.json
  tests/test_MODULE_016.py
  examples/example.py
  exports/__init__.py
```

## Status

- **Status:** ACTIVE
- **Coverage:** 100%
- **Tests:** `tests/test_MODULE_016.py`
- **Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF
