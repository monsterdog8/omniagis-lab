# MODULE_018_CLASSIFICATION_MATRIX

**Source:** `classification_matrix_schema_v1.json`
**Version:** 1.0.0
**Type:** GOVERNANCE ARTIFACT / DATA COLLECTION TEMPLATE
**Status:** TEMPLATE__AWAITING_CANONICAL_PACKET
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF

## Description

Schema template for real auditor data collection. Defines the structure for 5 auditors classifying 8 claims (C01-C08) = 40 total classification entries. Specifies the label schema, C08 gate thresholds, and the Krippendorff alpha computation trigger for inter-rater reliability.

This is NOT a code module. It is a JSON template document that must be filled in by real auditors after the owner confirms packet C01-C08 as CANONICAL_v1.

## Auditors

| Auditor ID | Type | Blind |
|------------|------|-------|
| HUMAN_01 | HUMAN | true |
| HUMAN_02 | HUMAN | true |
| GPT | AI | true |
| CLAUDE | AI | true |
| GROK | AI | true |

## Label Schema

| Label | Meaning |
|-------|---------|
| `SUPPORTED` | Claim is supported by the evidence |
| `PARTIALLY_SUPPORTED` | Claim is partially supported |
| `UNSUPPORTED` | Claim is not supported by the evidence |
| `INSUFFICIENT_EVIDENCE` | Evidence is insufficient to classify |

**Confidence:** Integer in [0, 10]

**Required fields per entry:** `auditor_id`, `claim_id`, `label`, `confidence`, `rationale_brief`

## C08 Gate

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `alpha_threshold_minimum` | 0.667 | Minimum acceptable Krippendorff alpha |
| `alpha_threshold_substantial` | 0.800 | Substantial agreement threshold |

If Krippendorff alpha < 0.667, the following claims are blocked: `external_replication`, `scientific_validation`, `public_promotion`.

## Pipeline

1. Owner confirms packet C01-C08 as CANONICAL_v1
2. Fill `packet_reference.canonical = true`
3. 5 auditors independently classify all 8 claims (blind)
4. 40 entries collected in `classifications[]`
5. Compute Krippendorff alpha
6. Evaluate C08 gate (alpha >= 0.667?)
7. Record result in `computed_metrics` and `c08_gate.result`
8. Proceed to META_AUDIT

## Classification Entry Format

```json
{
  "auditor_id": "HUMAN_01",
  "claim_id": "C03",
  "claim_hash": "<sha256 of canonical claim text>",
  "label": "SUPPORTED",
  "confidence": 8,
  "rationale_brief": "The evidence in the repo supports this claim within its stated scope.",
  "timestamp": "2026-06-07T12:00:00Z"
}
```

## Files

```
MODULE_018_CLASSIFICATION_MATRIX/
  manifest.json
  README.md
  schema.json
  tests/test_MODULE_018.py
  examples/example.py
  exports/classification_matrix_schema_v1.json
```

## Status

- **Status:** TEMPLATE__AWAITING_CANONICAL_PACKET
- **Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF
- **Blocked until:** Owner confirms CANONICAL_v1 packet
