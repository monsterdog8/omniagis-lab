# MODULE_012_SCORE_REPORT

## Purpose

Scoring engine for canonical 7-section audit reports.

Scores an AuditReport-formatted markdown text against a case JSON oracle on 7 weighted components. All output scores are LOCAL_LAB_ONLY heuristics — not calibrated probabilities.

## Source

`gpts_core/score_report.py`

## Scoring Components (default weights, sum = 100)

| Component | Max Points | Description |
|---|---|---|
| `global_and_subverdicts` | 20 | Verdict exact match + sub-verdict coverage |
| `critical_points_coverage` | 20 | Coverage of required critical points |
| `confidence_calibration` | 15 | Closeness of confidence score to oracle |
| `discriminant_tests` | 15 | Count of discriminant tests in Detailed Analysis |
| `probative_separation` | 10 | Presence of all 5 evidence channels |
| `weakness_taxonomy` | 10 | Weakness micro-syntax field completeness |
| `fail_closed_and_limitations` | 10 | Fail-closed discipline in Limitations |

## Public API

| Symbol | Type | Description |
|---|---|---|
| `DEFAULT_ALLOWED_VERDICTS` | `List[str]` | 5 allowed verdict strings |
| `DEFAULT_REQUIRED_SECTIONS` | `List[str]` | 7 required section names |
| `DEFAULT_WEAKNESS_FIELDS` | `List[str]` | 5 required weakness fields |
| `CanonicalValidation` | dataclass | Canonical validation result |
| `ComponentScores` | dataclass | Per-component integer scores |
| `normalize_text` | function | Unicode-normalize and lowercase |
| `extract_sections` | function | Parse markdown heading sections |
| `extract_verdict` | function | Extract verdict string from text |
| `extract_confidence` | function | Extract confidence (x/10) from text |
| `score_verdicts` | function | Score verdict matching |
| `score_critical_points` | function | Score critical point coverage |
| `score_confidence` | function | Score confidence calibration |
| `score_discriminant_tests` | function | Count discriminant tests |
| `score_probative_separation` | function | Score 5-channel evidence separation |
| `score_weakness_taxonomy` | function | Score weakness micro-syntax |
| `score_fail_closed` | function | Score fail-closed discipline |
| `validate_case_json` | function | Validate case oracle JSON |
| `evaluate` | function | Full oracle-based evaluation |
| `score_audit_report` | function | Standalone report scoring |

## 5 Evidence Channels

For full `probative_separation` score, the report must mention all 5:
`observation`, `theory`, `modeling`, `interpretation`, `inference`

## Dependencies

All stdlib: `re`, `unicodedata`, `dataclasses`, `json`, `pathlib`

## Coverage

70%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All scores are local heuristic computations against keyword patterns. No external benchmark or calibration is claimed.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_012.py` — extracted from `tests/test_gpts_core.py::TestScoreReport`
