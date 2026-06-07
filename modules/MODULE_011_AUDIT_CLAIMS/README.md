# MODULE_011_AUDIT_CLAIMS

## Purpose

Scientific claim auditing with 7-section canonical audit reports.

Audits a text claim against a corpus of artifacts using keyword overlap, channel-aware evidence extraction (observation / theory / modeling / inference / interpretation), and a fail-closed verdict policy.

**Verdict scale:** SUPPORTED / PARTIALLY SUPPORTED / WEAKLY SUPPORTED / UNSUPPORTED / UNCERTAIN

**Confidence:** 0–10 heuristic score (not a calibrated probability).

All outputs: LOCAL_LAB_ONLY — not external proof.

## Source

`gpts_core/audit_claims.py`

## Public API

| Symbol | Type | Description |
|---|---|---|
| `VERDICTS` | `List[str]` | Ordered verdict scale (5 values) |
| `WEAKNESS_TYPES` | `List[str]` | 4 canonical weakness type names |
| `STATUSES` | `List[str]` | 3 weakness statuses |
| `SEVERITIES` | `List[str]` | 3 severity levels |
| `CAUSES` | `List[str]` | 5 probable cause categories |
| `STRICT_SECTIONS` | `List[str]` | 7 required section headings |
| `SourceRecord` | dataclass | Input artifact source record |
| `EvidenceSlice` | dataclass | Scored evidence sentence from a source |
| `Weakness` | dataclass | Audit weakness entry with validate() and render() |
| `PropositionAssessment` | dataclass | Assessed proposition with support score |
| `AuditReport` | dataclass | Full 7-section report with validate(), render_markdown(), to_dict() |
| `load_artifacts` | function | Load text from files/directories |
| `extract_evidence` | function | Extract and rank evidence sentences |
| `decompose_claim` | function | Split compound claim into propositions |
| `assess_proposition` | function | Score one proposition against evidence |
| `detect_logical_contradictions` | function | Scan for lexical contradiction pairs |
| `audit_claim` | function | Full audit pipeline returning AuditReport |
| `inspect_document` | function | Document inspection without explicit claim |
| `validate_canonical_report` | function | Check 7-section format compliance |
| `batch_audit` | function | Audit multiple claims in batch |

## 7 Required Sections

1. Title
2. Executive Summary
3. Detailed Analysis
4. Weakness Taxonomy
5. Limitations
6. Recommendations
7. Sources

## Dependencies

All stdlib: `re`, `pathlib`, `dataclasses`, `csv`, `json`, `math`, `statistics`

## Coverage

52%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All verdicts and confidence scores are local heuristic assessments. No external benchmark or calibration is claimed.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_011.py` — extracted from `tests/test_gpts_core.py::TestAuditClaims`
