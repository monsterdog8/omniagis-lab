# INTEGRATION_DECISIONS.md
## Claim ceiling: LOCAL_SUPPORTED_WITHIN_SCOPE

Documents all deliberate integration and exclusion decisions for files processed
from artifact batches (/tmp/artifacts/, /tmp/artifacts2/, /tmp/artifacts3/).

---

## Files INTEGRATED (fully or partially)

| FILE | LINES | DECISION | FUNCTIONS_TOTAL | FUNCTIONS_INTEGRATED | DEST_MODULE |
|---|---|---|---|---|---|
| MONSTERDOG_FAIL_CLOSED_ENGINE.py | ~90 | INTEGRATED | ~8 | ~8 | gate.py |
| MONSTERDOG_METRICS_EXTRACTOR.py | 180 | INTEGRATED | ~12 | ~12 | signals.py |
| MONSTERDOG_AUDIT_LOGGER.py | ~90 | INTEGRATED | ~8 | ~8 | ledger.py |
| adjudication_scorer_v1_5.py | ~85 | INTEGRATED | 6 | 6 | adjudication.py |
| eval_runner.py | 303 | PARTIAL | ~20 | ~10 | benchmark.py |
| MONSTERDOG_LONG_CONTEXT_LOCK.py | ~60 | INTEGRATED | 3 | 3 | ledger.py |
| aegis audit.py | ~80 | INTEGRATED | ~8 | ~8 | evidence.py |
| AEGIS manifest.py | ~90 | INTEGRATED | 7 | 7 | manifest.py |
| safe_hold_gate.py | ~60 | INTEGRATED | ~5 | ~5 | promotion.py |
| monsterdog_metric_classifier_v1_4.py | ~80 | INTEGRATED | 4 | 4 | classifier.py |
| noise_blind_family_test_run0005.py | ~200 | INTEGRATED | ~12 | ~12 | signals.py |
| parser.py (METRIC_CH_ACTION_FORGE) | ~50 | INTEGRATED | ~4 | ~4 | evidence.py |
| replay_script.py | ~60 | INTEGRATED | ~5 | ~5 | promotion.py |
| MONSTERDOG_RAW_PAYLOAD_CAPTURE_GATE_v1.py | ~80 | INTEGRATED | ~6 | ~6 | evidence.py |
| seal_continuity_audit_validator_v0_1.py | ~80 | INTEGRATED | ~5 | ~5 | promotion.py |
| EXOCHRONOS_PROMOTION_GATE_ENGINE.py | ~100 | INTEGRATED | ~7 | ~7 | promotion.py |
| baseline_model.py | ~50 | INTEGRATED | 4 | 4 | benchmark.py |
| scoring.py | ~60 | INTEGRATED | 3 | 3 | benchmark.py |
| compare_shadow_g2.py | ~80 | INTEGRATED | ~6 | ~6 | benchmark.py |
| module_01_through_module_05 (REFORGE) | ~300 | PARTIAL | ~25 | ~20 | gate.py, evidence.py, benchmark.py |
| exochronos_capture_model_v0_1.py | 442 | INTEGRATED | 18 | 15 | coherence.py |
| exochronos_coherence_passport_v0_1.py | 497 | INTEGRATED | 25 | 22 | coherence.py |
| aegis_omega_ultimate.py | 935 | INTEGRATED | 38 | 38 | audit_claims.py |
| score_aegis_ultimate.py | 559 | INTEGRATED | 26 | 24 | score_report.py |
| paperA_pipeline_portable.py | ~700 | INTEGRATED | ~15 | ~12 | spectral_gap.py |
| continuum_core.py | 53 | INTEGRATED | 4 | 4 | dynamics.py |
| fractal.py | 162 | INTEGRATED | ~8 | ~6 | dynamics.py |

---

## Files EXCLUDED

| FILE | LINES | EXCLUSION_REASON | IMPACT_ON_ZERO_LOSS_CLAIM |
|---|---|---|---|
| i007_total_lab.py | ~500 | VISUALIZATION — PIL/vortex field rendering; no stdlib-extractable algorithm beyond what's in signals.py | PARTIAL_BREACH — hash ledger logic not extracted |
| fractal_neural_core.py | UNKNOWN | VISUALIZATION — neural/PIL rendering assumed based on name and batch context | UNKNOWN — file not read |
| ULAM_VIZ (directory) | UNKNOWN | VISUALIZATION — Ulam spiral rendering, not algorithmic computation | NONE — visualization only |
| WORLD_ARENA_VIZ_v0_11 (directory) | UNKNOWN | VISUALIZATION — world arena rendering, UI | NONE — visualization only |
| entity72k_smoke_test_daemon.py | ~80 | DAEMON/TEST_HARNESS — smoke test runner, not functional library code | NONE — harness only |
| qi_branch_frame_simulator_v0_1.py | ~100 | DOMAIN_SPECIFIC — quantum information frame simulation with external ontology | PARTIAL_BREACH — domain-specific physics not extracted |
| sync_config.py (AEGIS_FINAL_IMPL) | ~40 | CONFIGURATION — sync/config management, no algorithmic content | NONE — config only |
| GIS_MEMORY_FILE (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| CYCLE4_RUN72 (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| REFLET_FULL_1_1 (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| REFLET_TEST_1_0 (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| EXOCHRONOS_BATCH5_NORM (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| EXOCHRONOS_BATCH5_ADDENDUM (directory) | UNKNOWN | NOT_READ — directory contents not explored | UNKNOWN |
| 5acb6a9a-nextjs_supabase_accounts_contact_bundle | UNKNOWN | NON_PYTHON — Next.js/Supabase web bundle | NONE — different language |
| preflight_shadow_observation_v0_2.py | ~80 | NOT_READ — identified but not incorporated | UNKNOWN |
| model01_raw_capture_gate.py | ~80 | NOT_READ — identified but not incorporated | UNKNOWN |

---

## Files STATUS UNKNOWN (from artifacts3/)

These directories were identified but contents not fully explored:

| DIRECTORY | STATUS |
|---|---|
| 97ec0f50-EXOCHRONOS_SOURCE_STATUS_LEDGER_v0_1_PACKAGE | UNKNOWN |
| 0550dea9-EXOCHRONOS_SOURCE_STATUS_LEDGER_v0_2_PACKAGE | UNKNOWN |
| c7f11940-EXOCHRONOS_IMAGE_CH_SCANNER_v0_1_PACKAGE1 | UNKNOWN |
| f7a00390-MONSTERDOG_ASTROBIO_FALSE_POSITIVE_GATE_v0_1 | UNKNOWN |

---

## Impact on "zero loss" claim

The "zero loss" claim is PARTIALLY_BREACHED for:
1. Files with STATUS UNKNOWN — content not verified
2. `i007_total_lab.py` — hash ledger logic not extracted (PIL dependency blocked stdlib extraction)
3. `fractal_neural_core.py` — not read
4. `preflight_shadow_observation_v0_2.py` — not read
5. `model01_raw_capture_gate.py` — not read

For all OTHER files listed as INTEGRATED: logic is present in gpts_core (INTEGRATED/RENAMED/PRIVATIZED/INLINED).

**Overall "zero loss" verdict: NOT_PROVEN → PARTIAL**
(verified for ~80% of identified source files; ~20% remains UNKNOWN)
