# UMP-001 — IMPLEMENTATION REPORT 001

## Scope
Patch-forward implementation only. Stage0 remains immutable. No real GTAB trial was executed. The Stage0 branch pointer has been restored to the frozen Stage0 commit; all child work now lives on a separate child branch.

## Parent binding
- Stage0 capsule SHA-256: `3d000c5ec1ec27d1fde68fa528ac25e9d5c34b698e3cc7f684682cf180f3057c`
- Stage0 external preservation commit: `b91e808a76284828ded87e383febc16c6a01ac27`
- Claim ceiling: `STAGE0_PROTOCOL_FREEZE_ONLY`
- RFC3161: `NOT_OBTAINED`

## Branch topology
- Frozen parent branch: `ump-001-stage0-freeze` → `b91e808a76284828ded87e383febc16c6a01ac27`
- Patch-forward child branch: `ump-001-gtab-bind-001`
- Stage0 bytes/commit were not rewritten.

## Child artifacts implemented
- `GTAB_BIND_001.json` — `GTAB_RC3_001`, status `PROPOSED_NOT_EXECUTED`
- `UMP_001_HANGAR_SYNC_001.json`
- `ump_rc3_core.py`
- `tests/test_ump_rc3_core.py`
- `UMP_001_RC3_ENGINEERING_PILOT_CONFIG_V1.json`
- `ump_rc3_power_plan.py`
- `validate_ump_child.py`
- `.github/workflows/ump-001-child-selftest.yml`

## C2/C4 separation
- C2: ordinary RC world-model test over externally controlled depth/horizon settings.
- C4: one preregistered D* × Delta* post-commit randomized kill-shot.
- C2 success never promotes C4.

## C4 primary endpoint
Primary: proper-score skill versus the exact randomized null.
Secondary: `OMEGA_U`.

## GitHub Actions execution receipt
Workflow run: `36284553623`
Validated CI head: `d17df7181fe89005e103dc2876b4e07c7dc86757`
Child branch retarget commit: `d8b6be1e30889b47ab73b741c8f950dd37754382`
Runtime: CPython 3.13.15
Conclusion: `SUCCESS`

Executed checks:
1. Stage0 parent hash receipt unchanged.
2. Eight RC3 child tests PASS.
3. Commit/reveal round-trip PASS.
4. Three-way rejection boundary PASS.
5. RC boundary/asymptote checks PASS.
6. Proper-score null and confident-wrong penalty PASS.
7. Child structural validator PASS.
8. Power planner remains fail-closed without pilot SD and epsilon:
   - `status = BLOCKED_MISSING_PILOT_SD_AND_EPSILON`
   - `canonical_confirmatory_N = NOT_COMPUTED`

## Outside Agent / Hangar synchronization
Agent: `MONSTERDOG Ω∞ — MASTER CHAT KER ORCHESTRATOR 001`
Agent id: `df5d0695-dc6a-498d-aa0f-f89de6559e56`

Applied:
- five UMP must-intents,
- five UMP knowledge sources,
- C2/C4 firewall,
- Stage0 immutability rule,
- proper-score primary endpoint rule,
- GTAB_RC3 TBD blocker rule,
- anomaly→replication doctrine,
- five generated scenarios: 5/5 replies, 0 failures,
- four linked knowledge-base evals authored.

Live eval/review remains blocked by creator credits. No publish was attempted.

## Buildy
Buildy app/list operations returned internal tool errors during this implementation pass.
No Buildy app or persisted Buildy state is claimed.
GitHub + Outside Agent remain the verified external Hangar layers for this run.

## Current scientific state
- GTAB_REAL = NOT_BOUND
- OBSERVATION_DEVICE = TBD
- PRECISION = TBD
- SAMPLING_RATE = TBD
- R_OHM = TBD
- C_FARAD = TBD
- VOLTAGE_LEVELS = TBD
- D_PRIMARY = TBD
- DELTA_PRIMARY = TBD
- POSTCOMMIT_PHYSICAL_INDEPENDENCE = NOT_PROVEN
- EMPIRICAL_C2_EVIDENCE = 0
- EMPIRICAL_C4_EVIDENCE = 0
- OMEGA_U = NOT_MEASURED
- UNIVERSAL_CONNECTION = NOT_SUPPORTED

## Next gate
`GTAB_RC3_001_REAL_HARDWARE_BIND`

Bind the actual low-voltage RC device and independently measured observation path. Do not run the engineering pilot until R, C, voltage levels, observation device, precision, sampling rate, D*, Delta*, acquisition path, and reset/intervention timings are measured and frozen.

## Claim ceiling
`BINDING_SPEC_AND_HANGAR_INTEGRATION_ONLY`
