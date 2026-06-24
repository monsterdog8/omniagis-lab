"""Claim classification and evidence scoring — unified gate.

Combines gpts_core evidence gating (BLOCKED/BOUNDED/UNKNOWN text classification,
weighted evidence scoring) with phase_omega ValidationPipeline bridges
(validate_metric, classify_claim_scientific).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Claim classifier (text-based, gpts_core origin)
# ---------------------------------------------------------------------------

_FORBIDDEN: List[Tuple[str, str]] = [
    ("GLOBAL_SUPERIORITY",
     r"\b(top\s*1|best\s+in\s+the\s+world|beats?\s+all|beats\s+GPT|beats\s+Gemini|"
     r"supériorité globale|gagne\s+contre\s+tout)\b"),
    ("PRODUCTION_READY",
     r"\b(production[- ]ready|approved\s+for\s+production|prod\s+ready|"
     r"déploiement\s+réel)\b"),
    ("SCIENTIFIC_VALIDATION",
     r"\b(scientifically\s+validated|scientific\s+proof|validé\s+scientifiquement|"
     r"preuve\s+scientifique)\b"),
    ("CONSCIOUSNESS",
     r"\b(consciousness\s+proven|sentience\s+proven|conscience\s+prouvée|"
     r"être\s+vivant|âme\s+numérique)\b"),
    ("BENCHMARK_WON",
     r"\b(SOTA|state\s+of\s+the\s+art|benchmark\s+winner|benchmark\s+gagné)\b"),
    ("METAPHYSICS_AS_PROOF",
     r"\b(telepathy\s+proven|télépathie\s+prouvée|FTL\s+prouvé|"
     r"message\s+vers\s+le\s+passé)\b"),
]

_BOUNDED: List[Tuple[str, str]] = [
    ("LAB_HYPOTHESIS",
     r"\b(hypothesis|hypothèse|might|could|peut|pourrait|testable|bounded|borné|"
     r"LAB_ONLY|prototype|simulation)\b"),
    ("EVIDENCE_WORKFLOW",
     r"\b(raw|scoring|replay|hash|metric|data|independent[_\s]review|sidecar)\b"),
    ("ACTION_STRUCTURE",
     r"\b(structure|test|analyze|observer|action|30\s*minutes|diagnostic)\b"),
]


class ClaimStatus(str, Enum):
    BLOCKED = "BLOCKED"
    ALLOWED_BOUNDED = "ALLOWED_BOUNDED"
    UNKNOWN = "UNKNOWN_REQUIRES_REVIEW"


@dataclass(frozen=True)
class ClaimResult:
    text: str
    status: str
    reason: str
    hits: List[str]
    required_next_gate: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_claim(text: str) -> ClaimResult:
    """Classify a text claim as BLOCKED / ALLOWED_BOUNDED / UNKNOWN."""
    text = text or ""
    blocked = [name for name, pat in _FORBIDDEN if re.search(pat, text, re.I)]
    if blocked:
        return ClaimResult(
            text=text,
            status=ClaimStatus.BLOCKED.value,
            reason="Strong claim detected without full evidence (raw+scoring+replay+independent review).",
            hits=blocked,
            required_next_gate="RAW_COLLECTION + SCORING + REPLAY + INDEPENDENT_REVIEW",
        )
    bounded = [name for name, pat in _BOUNDED if re.search(pat, text, re.I)]
    if bounded:
        return ClaimResult(
            text=text,
            status=ClaimStatus.ALLOWED_BOUNDED.value,
            reason="Prudent formulation allowed as bounded hypothesis or local diagnostic.",
            hits=bounded,
            required_next_gate="RAW_CAPTURE_IF_CLAIM_STRENGTH_INCREASES",
        )
    return ClaimResult(
        text=text,
        status=ClaimStatus.UNKNOWN.value,
        reason="No forbidden pattern detected but scope is not explicitly bounded.",
        hits=[],
        required_next_gate="MANUAL_REVIEW_BEFORE_PUBLIC_LANGUAGE",
    )


# ---------------------------------------------------------------------------
# Evidence scoring (gpts_core origin)
# ---------------------------------------------------------------------------

EVIDENCE_WEIGHTS: Dict[str, float] = {
    "DATA": 0.20,
    "RAW": 0.20,
    "SCORING": 0.20,
    "REPLAY": 0.15,
    "INDEPENDENCE": 0.15,
    "SAFETY": 0.10,
}

MATURITY_STAGES = ["IDEA", "DESIGN", "PROTOTYPE", "RAW", "SCORING", "REPLAY", "REVIEW", "DEPLOY"]


def _clip(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class EvidenceInput:
    DATA: float = 0.0
    RAW: float = 0.0
    SCORING: float = 0.0
    REPLAY: float = 0.0
    INDEPENDENCE: float = 0.0
    SAFETY: float = 0.0
    raw_expected: int = 0
    raw_valid: int = 0
    scoring_done: bool = False
    replay_available: bool = False
    independent_review: bool = False
    strong_public_claim: bool = False


@dataclass(frozen=True)
class EvidenceScore:
    dimensions: Dict[str, float]
    weighted_raw: float
    penalties: List[str]
    final_score: float
    scoring_status: str
    public_claim_allowed: bool
    verdict: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_evidence_score(inp: EvidenceInput) -> EvidenceScore:
    """Compute weighted evidence score with fail-closed penalties."""
    dims = {k: _clip(getattr(inp, k)) for k in EVIDENCE_WEIGHTS}
    weighted = round(sum(dims[k] * EVIDENCE_WEIGHTS[k] for k in EVIDENCE_WEIGHTS), 6)
    penalties: List[str] = []
    notes: List[str] = []

    raw_pass = bool(inp.raw_expected and inp.raw_valid == inp.raw_expected)

    if inp.raw_expected and inp.raw_valid < inp.raw_expected:
        penalties.append("-1.00:RAW_COLLECTION_INCOMPLETE")
    if dims["RAW"] <= 0.0:
        penalties.append("-1.00:NO_RAW_OUTPUTS")
    if inp.strong_public_claim:
        penalties.append("-0.40:STRONG_CLAIM_WITHOUT_FULL_EVIDENCE")
    if not inp.scoring_done:
        penalties.append("-0.25:SCORING_NOT_EXECUTED")
    if not inp.replay_available:
        penalties.append("-0.15:REPLAY_NOT_ESTABLISHED")
    if dims["INDEPENDENCE"] <= 0.0 or not inp.independent_review:
        penalties.append("-0.15:NO_INDEPENDENT_REVIEW")
    if dims["SAFETY"] < 0.5:
        penalties.append("-0.10:SAFETY_INSUFFICIENT")

    if raw_pass and not inp.scoring_done:
        notes.append("Raw gate passed; scoring can be initiated but not claimed.")
    elif not raw_pass:
        notes.append("Raw gate not passed; scoring remains blocked.")

    penalty_val = sum(abs(float(p.split(":", 1)[0])) for p in penalties)
    final = max(0.0, min(1.0, round(weighted - penalty_val, 6)))

    scoring_status = (
        "READY_FOR_SCORING" if raw_pass and not inp.scoring_done
        else ("SCORING_DONE_LOCAL" if raw_pass and inp.scoring_done else "BLOCKED")
    )
    public_ok = bool(
        final >= 0.85
        and dims["RAW"] > 0.0
        and dims["SCORING"] > 0.0
        and dims["REPLAY"] > 0.0
        and dims["INDEPENDENCE"] > 0.0
        and dims["SAFETY"] >= 0.75
        and inp.independent_review
    )

    return EvidenceScore(
        dimensions=dims,
        weighted_raw=weighted,
        penalties=penalties,
        final_score=final,
        scoring_status=scoring_status,
        public_claim_allowed=public_ok,
        verdict="CLAIM_REVIEW_READY_NOT_PUBLIC_PROOF" if public_ok else "BLOCKED_FAIL_CLOSED",
        notes=notes,
    )


def from_gate_counts(
    expected: int,
    present: int,
    valid: int,
    strict_sidecars: int = 0,
    safety: float = 0.7,
    scoring: bool = False,
    replay: bool = False,
    independence: bool = False,
    strong_public_claim: bool = False,
) -> EvidenceInput:
    """Construct EvidenceInput from raw output counts."""
    expected = max(0, int(expected or 0))
    valid = max(0, int(valid or 0))
    strict_sidecars = max(0, int(strict_sidecars or 0))
    data = 1.0 if expected > 0 else 0.0
    raw_ratio = valid / expected if expected else 0.0
    sidecar_ratio = strict_sidecars / expected if expected and strict_sidecars else raw_ratio
    raw = _clip(min(raw_ratio, sidecar_ratio))
    return EvidenceInput(
        DATA=data, RAW=raw,
        SCORING=1.0 if scoring else 0.0,
        REPLAY=1.0 if replay else 0.0,
        INDEPENDENCE=1.0 if independence else 0.0,
        SAFETY=_clip(safety),
        raw_expected=expected,
        raw_valid=valid,
        scoring_done=scoring,
        replay_available=replay,
        independent_review=independence,
        strong_public_claim=strong_public_claim,
    )


def maturity_map(**flags: bool) -> Dict[str, Any]:
    """Track invention maturity. Public claim rights remain BLOCKED regardless of maturity."""
    stages = {s: bool(flags.get(s.lower(), False)) for s in MATURITY_STAGES}
    highest = "NONE"
    for s in MATURITY_STAGES:
        if stages[s]:
            highest = s
        else:
            break
    return {
        "stages": stages,
        "highest_maturity": highest,
        "public_claim_right": "BLOCKED",
        "production_status": "LOCKED",
    }


def proof_firewall(dims: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Final public-claim gate. Returns BLOCKED unless all proof dimensions are non-zero."""
    dims = dims or {}
    required = list(EVIDENCE_WEIGHTS.keys())
    missing = [k for k in required if _clip(dims.get(k, 0.0)) <= 0.0]
    if missing:
        return {"verdict": "BLOCKED_FAIL_CLOSED", "reason": "Proof dimensions missing", "missing": missing}
    if _clip(dims.get("SAFETY", 0.0)) < 0.75:
        return {"verdict": "BLOCKED_FAIL_CLOSED", "reason": "SAFETY below 0.75", "missing": []}
    return {
        "verdict": "READY_FOR_INDEPENDENT_REVIEW_NOT_PUBLIC_PROOF",
        "reason": "All local dimensions non-zero; independent external review still required.",
        "missing": [],
    }


# ---------------------------------------------------------------------------
# Scientific gate bridges (phase_omega ValidationPipeline origin)
# ---------------------------------------------------------------------------

def validate_metric(
    metric_name: str,
    formula_id: str,
    result: float,
    r_squared: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate a metric result against R² acceptance criteria.

    R² >= 0.95 → PASS | >= 0.80 → CONDITIONAL | < 0.80 → FAIL | nan/inf → INVALID | no R² → UNKNOWN
    """
    import math as _math
    verdict: Dict[str, Any] = {
        "metric": metric_name,
        "formula_id": formula_id,
        "result": result,
        "r_squared": r_squared,
        "status": "UNKNOWN",
    }
    if not isinstance(result, float) or _math.isnan(result) or _math.isinf(result):
        verdict["status"] = "INVALID"
        return verdict
    if r_squared is not None:
        if r_squared >= 0.95:
            verdict["status"] = "PASS"
        elif r_squared >= 0.80:
            verdict["status"] = "CONDITIONAL"
        else:
            verdict["status"] = "FAIL"
    return verdict


def classify_claim_scientific(
    observation: Optional[str],
    evidence_r2: Optional[float] = None,
    theory_support: bool = False,
    causality: bool = False,
) -> str:
    """Classify scientific claim using fail-closed R² rules.

    Returns: UNKNOWN | OBSERVED | SUPPORTED | PLAUSIBLE | REFUTED
    """
    if observation is None:
        return "UNKNOWN"
    if evidence_r2 is not None and evidence_r2 >= 0.95:
        return "OBSERVED"
    if evidence_r2 is not None and evidence_r2 >= 0.80 and theory_support:
        return "SUPPORTED"
    if evidence_r2 is not None and evidence_r2 >= 0.60:
        return "PLAUSIBLE"
    if evidence_r2 is not None and evidence_r2 < 0:
        return "REFUTED"
    return "UNKNOWN"
