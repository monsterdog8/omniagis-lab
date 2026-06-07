"""MODULE_012_SCORE_REPORT — runnable usage examples.

Demonstrates: extract_sections, extract_verdict, extract_confidence,
score_fail_closed, score_audit_report, and evaluate with a case oracle.
"""
from __future__ import annotations

from gpts_core.score_report import (
    extract_sections,
    extract_verdict,
    extract_confidence,
    score_fail_closed,
    score_probative_separation,
    score_weakness_taxonomy,
    score_audit_report,
    evaluate,
    validate_case_json,
    DEFAULT_ALLOWED_VERDICTS,
    DEFAULT_REQUIRED_SECTIONS,
    DEFAULT_WEAKNESS_FIELDS,
)

# ---------------------------------------------------------------------------
# Example 1: Parse a markdown report into sections and extract key fields
# ---------------------------------------------------------------------------

report_md = (
    "## 1. Title\nTest Report\n"
    "## 2. Executive Summary\n"
    "The claim is weakly supported. Verdict: WEAKLY SUPPORTED\n"
    "confidence score: 4.5/10\n"
    "## 3. Detailed Analysis\n"
    "Evidence shows partial support via observation and inference. "
    "Theoretical modeling supports the framework. Interpretation is limited.\n"
    "## 4. Weakness Taxonomy\n"
    "Type: Explanatory Limit\nStatus: Established\nSeverity: Medium\n"
    "Probable Cause: Modeling\nDiscriminant Test: Provide benchmark data.\n"
    "## 5. Limitations\n"
    "The analysis is limited to available artifacts. Conclusions are uncertain and fragile.\n"
    "## 6. Recommendations\nCollect independent data.\n"
    "## 7. Sources\nArtifact A.\n"
)

sections = extract_sections(report_md)
print(f"[Example 1] Parsed sections: {list(sections.keys())}")

verdict = extract_verdict(report_md, DEFAULT_ALLOWED_VERDICTS)
print(f"[Example 1] Extracted verdict: {verdict}")

confidence = extract_confidence(report_md)
print(f"[Example 1] Extracted confidence: {confidence}/10")

# ---------------------------------------------------------------------------
# Example 2: Score individual components
# ---------------------------------------------------------------------------

# Fail-closed discipline
score_fc, note_fc = score_fail_closed(sections)
print(f"\n[Example 2] score_fail_closed: {score_fc}/10 — {note_fc}")

# Probative separation (5 evidence channels)
score_ps, note_ps = score_probative_separation(report_md)
print(f"[Example 2] score_probative_separation: {score_ps}/10 — {note_ps}")

# Weakness taxonomy fields
score_wt, note_wt = score_weakness_taxonomy(sections, DEFAULT_WEAKNESS_FIELDS)
print(f"[Example 2] score_weakness_taxonomy: {score_wt}/10 — {note_wt}")

# ---------------------------------------------------------------------------
# Example 3: Standalone report scoring with score_audit_report (no oracle required)
# ---------------------------------------------------------------------------

result = score_audit_report(report_md)
print(f"\n[Example 3] score_audit_report total: {result['total']}/100")
print(f"[Example 3] canonical passed: {result['canonical_validation']['passed']}")
scores = result["scores"]
print(f"[Example 3] Component scores:")
for k, v in scores.items():
    print(f"  {k}: {v}")
print(f"[Example 3] claim_ceiling: {result['claim_ceiling']}")

# Empty report
result_empty = score_audit_report("")
print(f"\n[Example 3] Empty report total: {result_empty['total']}/100")
