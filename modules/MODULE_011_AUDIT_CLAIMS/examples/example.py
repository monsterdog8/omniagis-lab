"""MODULE_011_AUDIT_CLAIMS — runnable usage examples.

Demonstrates: decompose_claim, audit_claim, validate_canonical_report,
batch_audit, and inspect_document.
"""
from __future__ import annotations

from gpts_core.audit_claims import (
    decompose_claim,
    audit_claim,
    validate_canonical_report,
    batch_audit,
    inspect_document,
    VERDICTS,
    STRICT_SECTIONS,
)

# ---------------------------------------------------------------------------
# Example 1: Decompose a compound claim into testable propositions
# ---------------------------------------------------------------------------

claim = "The system is coherent, traceable, and operationally validated."
props = decompose_claim(claim)
print("[Example 1] decompose_claim:")
for i, p in enumerate(props, 1):
    print(f"  {i}. {p}")

# ---------------------------------------------------------------------------
# Example 2: Audit a claim without external files (conversational mode)
# ---------------------------------------------------------------------------

report = audit_claim(
    claim="The analysis protocol is fully validated and reproducible.",
    title="Lab Protocol Audit",
    inputs=[],  # no corpus files — will use conversational source
    add_conversation_source=True,
)

print(f"\n[Example 2] Audit report verdict: {report.verdict}")
print(f"[Example 2] Confidence: {report.confidence:.1f}/10")
print(f"[Example 2] Propositions assessed: {len(report.propositions)}")
print(f"[Example 2] Weaknesses: {len(report.weaknesses)}")
print(f"[Example 2] Limitations: {len(report.limitations)}")

# Serialize to dict (includes claim_ceiling)
d = report.to_dict()
print(f"[Example 2] claim_ceiling: {d.get('claim_ceiling')}")
print(f"[Example 2] production_unlocked: {d.get('production_unlocked', False)}")

# ---------------------------------------------------------------------------
# Example 3: Validate a canonical report's 7-section structure
# ---------------------------------------------------------------------------

# A minimal valid 7-section report
report_md = "\n".join([
    "## 1. Title",
    "Test Report",
    "## 2. Executive Summary",
    "The claim is weakly supported. Verdict: WEAKLY SUPPORTED. Confidence score: 4.0/10",
    "## 3. Detailed Analysis",
    "Evidence shows partial support via observation and inference.",
    "## 4. Weakness Taxonomy",
    "Type: Explanatory Limit\nStatus: Established\nSeverity: Medium\n"
    "Probable Cause: Modeling\nDiscriminant Test: Provide additional benchmark data.",
    "## 5. Limitations",
    "Analysis is limited to provided artifacts. Conclusions are uncertain.",
    "## 6. Recommendations",
    "Collect independent benchmark results.",
    "## 7. Sources",
    "Artifact A.",
])

is_valid, errors = validate_canonical_report(report_md)
print(f"\n[Example 3] validate_canonical_report: valid={is_valid}")
if errors:
    print(f"[Example 3] Errors: {errors}")
else:
    print("[Example 3] All 7 sections present, no missing weakness fields.")

# A report missing sections
is_valid_short, errors_short = validate_canonical_report("just a claim with no sections")
print(f"[Example 3] Short text valid={is_valid_short}, errors count={len(errors_short)}")
print(f"[Example 3] Required sections: {STRICT_SECTIONS}")
