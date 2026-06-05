# EQUIVALENCE_REPORT.md
## Claim ceiling: LOCAL_MEASURED — reproducible on this machine; not external proof

**Date:** 2026-06-05
**Command:** `pytest tests/test_equivalence_coherence.py tests/test_equivalence_benchmark.py -v`
**Result:** 21 passed, 0 failed

---

## Source mapping

| SOURCE_FILE | SOURCE_FUNCTION | DEST_MODULE | DEST_FUNCTION | INTERFACE_DELTA |
|---|---|---|---|---|
| `exochronos_capture_model_v0_1.py` | `shannon_entropy` | `coherence.py` | `shannon_entropy` | RENAMED_SIG: takes discrete int symbols; source takes float probs |
| `exochronos_capture_model_v0_1.py` | `joint_entropy` | `coherence.py` | `joint_entropy` | IDENTICAL |
| `exochronos_capture_model_v0_1.py` | `digitize` | `coherence.py` | `digitize` | IDENTICAL |
| `exochronos_capture_model_v0_1.py` | `global_coherence` | `coherence.py` | (inlined in `compute_metric_fields`) | INLINED |
| `scoring.py` (monsterdog_kaggle_minibench) | `calc_weighted_stability` | `benchmark.py` | `mse_score` | RENAMED — same formula |

---

## Per-function equivalence results

### Module: `gpts_core/coherence.py`

| FUNCTION | TEST_CLASS | TESTS_RUN | TESTS_PASS | TESTS_FAIL | STATUS |
|---|---|---|---|---|---|
| `shannon_entropy` | `TestShannonEntropyEquivalence` | 5 | 5 | 0 | VERIFIED |
| `joint_entropy` | `TestJointEntropyEquivalence` | 3 | 3 | 0 | VERIFIED |
| `digitize` | `TestDigitizeEquivalence` | 3 | 3 | 0 | VERIFIED |
| `global_coherence` (inlined) | `TestGlobalCoherenceEquivalence` | 1 | 1 | 0 | VERIFIED |

**Subtotal coherence:** 12/12 PASS

### Interface delta note — `shannon_entropy`

The source function (`exochronos_capture_model_v0_1.py`) takes a list of float probabilities summing to 1.
The destination (`gpts_core/coherence.py`) takes a sequence of discrete integer symbols and computes
frequency-based probabilities internally.

Both compute `H = -Σ p_i · log2(p_i)`. Equivalence tests verify identical numerical output when the
same empirical distribution is expressed as symbol sequences. This is a RENAMED_SIG, not a semantic loss.

### Module: `gpts_core/benchmark.py`

| FUNCTION | TEST_CLASS | TESTS_RUN | TESTS_PASS | TESTS_FAIL | STATUS |
|---|---|---|---|---|---|
| `mse_score` ↔ `calc_weighted_stability` | `TestMseScoreEquivalence` | 9 | 9 | 0 | VERIFIED |

**Test cases:**

| TEST_NAME | INPUT | EXPECTED | STATUS |
|---|---|---|---|
| `test_perfect_predictions` | y_true={A:1.0}, y_pred={A:1.0} | score=1.0 | PASS |
| `test_all_missing` | y_pred has no matching keys | score=0.0 | PASS |
| `test_empty_truth` | y_true={} | score=0.0 | PASS |
| `test_max_error` | diff=1.0 per key | score=0.0 | PASS |
| `test_single_perfect` | one key, perfect match | score=1.0 | PASS |
| `test_invalid_prediction` | NaN prediction | score=0.0 | PASS |
| `test_out_of_range_prediction` | pred outside [0,1] | score=0.0 | PASS |
| `test_mixed_keys` | partial overlap | 0.0 ≤ score ≤ 1.0 | PASS |
| `test_multiple_samples` | 3 keys, various errors | 0.0 ≤ score ≤ 1.0 | PASS |

**Subtotal benchmark:** 9/9 PASS

---

## Grand total

| SCOPE | TESTS_RUN | TESTS_PASS | TESTS_FAIL | VERDICT |
|---|---|---|---|---|
| coherence functions | 12 | 12 | 0 | VERIFIED |
| benchmark functions | 9 | 9 | 0 | VERIFIED |
| **TOTAL** | **21** | **21** | **0** | **VERIFIED** |

---

## Functions NOT equivalence-tested (and why)

| FUNCTION | REASON |
|---|---|
| `audit_claim`, `decompose_claim`, etc. | Source (`aegis_omega_ultimate.py`) has no pure-math counterpart to diff against; logic is heuristic text processing, not numeric formula |
| `score_audit_report` and subfunctions | Same: heuristic scoring, no numerical ground truth in source to compare |
| `pm_map`, `ulam_matrix`, `canonical_gap` | No stdlib source equivalent to compare; source (`paperA_pipeline_portable.py`) uses numpy throughout; equivalence would be numpy vs numpy, redundant |
| `ContinuumState`, `FractalEngine` | Stochastic (noise injection); deterministic equivalence test not applicable |
| `classify_claim`, `compute_evidence_score` | Source (`MONSTERDOG_FAIL_CLOSED_ENGINE.py`) was integrated by restructuring; no verbatim function to compare |

---

## Claim ceiling per function

| FUNCTION | CLAIM_CEILING |
|---|---|
| `shannon_entropy` | LOCAL_MEASURED — 5 deterministic unit tests, 21 total in suite |
| `joint_entropy` | LOCAL_MEASURED — 3 deterministic unit tests |
| `digitize` | LOCAL_MEASURED — 3 deterministic unit tests |
| `mse_score` | LOCAL_MEASURED — 9 deterministic unit tests vs source formula |
| All other functions | LOCAL_SUPPORTED_WITHIN_SCOPE — behavioral tests only, no source diff |

---

## Reproducibility

```bash
cd /home/user/omniagis-lab
pytest tests/test_equivalence_coherence.py tests/test_equivalence_benchmark.py -v
# Expected: 21 passed, 0 failed
```
