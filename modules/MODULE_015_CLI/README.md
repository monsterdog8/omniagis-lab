# MODULE_015_CLI

**Source:** `gpts_core/cli.py`
**Version:** 1.0.0
**Coverage:** 0% (subprocess-only interface, not unit-tested)
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF

## Description

Unified 14-subcommand CLI providing a single `gpts-core` entry point that dispatches to all other gpts_core modules. Each subcommand runs the corresponding module function and writes JSON (or Markdown) to stdout. Exit codes signal pass/fail for use in shell pipelines.

## Subcommands

| Subcommand | Description | Exit 0 | Exit 2 |
|------------|-------------|--------|--------|
| `analyze-signal` | Compute structure metrics for a CSV signal column | always | no numeric values found |
| `classify-metric` | Classify a metric row | always | — |
| `validate-records` | Validate a JSONL file of raw output records | pass_gate | not pass_gate |
| `classify-claim` | Classify claim as BLOCKED/BOUNDED/UNKNOWN | not BLOCKED | BLOCKED |
| `evidence-score` | Compute evidence score from gate counts | public_claim_allowed | not allowed |
| `replay-ledger` | Replay JSONL event ledger (hash chain check) | status=PASS | status!=PASS |
| `seal-audit` | Validate SEAL continuity JSONL ledger | all_valid | not all_valid |
| `promote` | Evaluate promotion candidates from CSV matrices | candidates found | no candidates |
| `build-manifest` | Build file manifest for a directory | always | — |
| `adjudicate` | Score a manual adjudication CSV | always | — |
| `coherence-passport` | Build and validate a coherence passport | replay_status=PASS | FAIL |
| `audit-claim` | Audit a text claim (7-section report) | always | — |
| `score-report` | Score a canonical audit report markdown | total >= min-score | total < min-score |
| `spectral-gap` | Compute Ulam spectral gap over alpha grid | regression_fail=None | regression_fail set |

## Public API

```python
build_parser() -> argparse.ArgumentParser
main(argv: list[str] | None = None) -> int
```

## Usage Examples

```bash
# Analyze a CSV signal
gpts-core analyze-signal data.csv --column value --sr 256.0 --pretty

# Classify a claim
gpts-core classify-claim "This model is production-ready" --pretty

# Run spectral gap pipeline
gpts-core spectral-gap --alpha-min 0.02 --alpha-max 0.20 --n-alpha 12 --n-bins 80 --n-traj 20000 --pretty

# Compute evidence score
gpts-core evidence-score --expected 10 --present 10 --valid 10 --safety 1.0 --scoring --replay --independence

# Audit a claim
gpts-core audit-claim --claim "The system achieves 99% coherence" --markdown
```

## Dependencies

- **stdlib:** `argparse`, `json`, `sys`, `pathlib`, `csv`
- **gpts_core modules:** signals, classifier, evidence, gate, ledger, promotion, manifest, adjudication, coherence, audit_claims, score_report, spectral_gap
- **third_party:** `numpy` (spectral-gap subcommand only)

## Files

```
MODULE_015_CLI/
  manifest.json
  README.md
  schema.json
  tests/test_MODULE_015.py
  examples/example.py
  exports/cli.py
```

## Status

- **Status:** ACTIVE
- **Coverage:** 0% (subprocess-only interface)
- **Tests:** `tests/test_MODULE_015.py`
- **Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF
