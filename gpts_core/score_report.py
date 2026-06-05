"""Scoring engine for canonical 7-section audit reports.

Scores an AuditReport-formatted markdown text against a case JSON oracle on
7 components: verdict matching, critical-point coverage, confidence calibration,
discriminant tests, probative separation, weakness taxonomy, and fail-closed discipline.

All output scores are LOCAL_LAB_ONLY heuristics — not calibrated probabilities.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_ALLOWED_VERDICTS = [
    "SUPPORTED",
    "PARTIALLY SUPPORTED",
    "WEAKLY SUPPORTED",
    "UNSUPPORTED",
    "UNCERTAIN",
]

DEFAULT_REQUIRED_SECTIONS = [
    "Title",
    "Executive Summary",
    "Detailed Analysis",
    "Weakness Taxonomy",
    "Limitations",
    "Recommendations",
    "Sources",
]

DEFAULT_WEAKNESS_FIELDS = [
    "Type",
    "Status",
    "Severity",
    "Probable Cause",
    "Discriminant Test",
]

_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "au", "aux",
    "en", "dans", "sur", "par", "pour", "avec", "sans", "que", "qui", "dont",
    "comme", "plus", "moins", "est", "sont", "a", "ont", "d", "l", "y", "ne",
    "pas", "se", "sa", "ses", "son", "the", "of", "to", "for", "on",
    "in", "by", "an", "at", "from",
}

_SECTION_ALIASES = {
    "title": "Title",
    "executive summary": "Executive Summary",
    "detailed analysis": "Detailed Analysis",
    "weakness taxonomy": "Weakness Taxonomy",
    "limitations": "Limitations",
    "recommendations": "Recommendations",
    "sources": "Sources",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CanonicalValidation:
    passed: bool
    missing_sections: List[str]
    format_issues: List[str]


@dataclass
class ComponentScores:
    global_and_subverdicts: int
    critical_points_coverage: int
    confidence_calibration: int
    discriminant_tests: int
    probative_separation: int
    weakness_taxonomy: int
    fail_closed_and_limitations: int


# ---------------------------------------------------------------------------
# Text normalization utilities
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("–", "-").replace("—", "-").replace("‑", "-")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_heading(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"^\d+[\.)]\s*", "", s)
    s = re.sub(r"^[ivxlcdm]+[\.)]\s*", "", s)
    s = re.sub(r"^\d+\s*-\s*", "", s)
    s = s.strip(" #:-")
    return _SECTION_ALIASES.get(s, s)


def _tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    tokens = re.findall(r"[a-z0-9µ]+", s)
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _keyword_set(s: str) -> set:
    return set(_tokenize(s))


def _semantic_match(candidate: str, text: str, threshold: float = 0.34) -> bool:
    cand_n = normalize_text(candidate)
    text_n = normalize_text(text)
    if cand_n in text_n:
        return True
    cand_tokens = _keyword_set(candidate)
    if not cand_tokens:
        return False
    present = sum(1 for tok in cand_tokens if tok in text_n)
    return (present / max(1, len(cand_tokens))) >= threshold


def _contains_nearby_verdict(
    text: str, anchor: str, expected_verdict: str, window: int = 400
) -> bool:
    text_n = normalize_text(text)
    anchor_n = normalize_text(anchor)
    verdict_n = normalize_text(expected_verdict)
    idx = text_n.find(anchor_n)
    if idx == -1:
        return False
    segment = text_n[max(0, idx - window): idx + len(anchor_n) + window]
    return verdict_n in segment


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = d
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def extract_sections(markdown: str) -> Dict[str, str]:
    """Parse markdown into section_name → body dict using heading detection."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

    for line in markdown.splitlines():
        m = heading_re.match(line)
        if m:
            current = _normalize_heading(m.group(2).strip())
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_verdict(text: str, allowed_verdicts: List[str] = DEFAULT_ALLOWED_VERDICTS) -> Optional[str]:
    """Extract the global verdict from report text."""
    allowed = sorted(allowed_verdicts, key=len, reverse=True)
    escaped = "|".join(re.escape(v) for v in allowed)
    m = re.search(rf"(?is)\bverdict\b\s*[:\-]?\s*({escaped})\b", text)
    return m.group(1).upper() if m else None


def extract_confidence(text: str) -> Optional[float]:
    """Extract confidence score (x/10) from report text."""
    for pattern in [
        r"(?is)\bconfidence score\b\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
        r"(?is)\bscore de confiance\b\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    ]:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


# ---------------------------------------------------------------------------
# Component scoring functions
# ---------------------------------------------------------------------------

def score_verdicts(
    report_text: str,
    report_sections: Dict[str, str],
    oracle: Dict[str, Any],
    allowed_verdicts: List[str] = DEFAULT_ALLOWED_VERDICTS,
    max_points: int = 20,
) -> Tuple[int, str, Dict[str, Any]]:
    """Score global + sub-verdict matching against oracle. Returns (score, note, details)."""
    overall = extract_verdict(report_text, allowed_verdicts)
    expected = oracle["global_verdict"]
    score_global = 10 if overall == expected else 0
    global_note = "Global verdict exact" if overall == expected else f"Global verdict mismatch (got={overall}, expected={expected})"

    sub_verdicts = oracle.get("sub_verdicts", [])
    if not sub_verdicts:
        score_sub, sub_note = 10, "No sub-verdicts required"
    else:
        covered = 0.0
        analysis_zone = "\n".join(report_sections.values())
        for sv in sub_verdicts:
            prop = sv.get("proposition", "")
            ev = sv.get("verdict", "")
            prop_match = _semantic_match(prop, analysis_zone, threshold=0.40)
            verdict_near = _contains_nearby_verdict(analysis_zone, prop, ev, window=500)
            covered += 1.0 if (prop_match and verdict_near) else (0.5 if prop_match else 0.0)
        score_sub = round(10 * covered / len(sub_verdicts))
        sub_note = f"Sub-verdict coverage = {covered:.1f}/{len(sub_verdicts)}"

    score = min(max_points, score_global + score_sub)
    return score, f"{global_note}; {sub_note}", {
        "actual_global_verdict": overall,
        "expected_global_verdict": expected,
        "score_global": score_global,
        "score_sub": score_sub,
    }


def score_critical_points(
    report_text: str,
    critical_points: List[str],
    max_points: int = 20,
) -> Tuple[int, str, List[str]]:
    """Score coverage of required critical points. Returns (score, note, uncovered)."""
    if not critical_points:
        return max_points, "No critical points required", []
    covered = 0.0
    uncovered: List[str] = []
    partial: List[str] = []
    for point in critical_points:
        if _semantic_match(point, report_text, threshold=0.42):
            covered += 1.0
        elif _semantic_match(point, report_text, threshold=0.25):
            covered += 0.5
            partial.append(point)
        else:
            uncovered.append(point)
    score = round(max_points * covered / len(critical_points))
    note = f"Critical points covered = {covered:.1f}/{len(critical_points)}"
    if partial:
        note += f"; partial={len(partial)}"
    return score, note, uncovered


def score_confidence(
    actual: Optional[float],
    expected: float,
    tolerance: Dict[str, float],
    max_points: int = 15,
) -> Tuple[int, str]:
    """Score confidence calibration. Returns (score, note)."""
    if actual is None:
        return 0, "Missing confidence score"
    diff = abs(actual - expected)
    if diff <= tolerance.get("full_score_if_abs_diff_lte", 0.5):
        return max_points, f"Full score (diff={diff:.2f})"
    if diff <= tolerance.get("light_penalty_if_abs_diff_lte", 1.0):
        return max_points - 2, f"Light penalty (diff={diff:.2f})"
    if diff <= tolerance.get("medium_penalty_if_abs_diff_lte", 2.0):
        return max_points - 5, f"Medium penalty (diff={diff:.2f})"
    return max(0, max_points - 10), f"Strong penalty (diff={diff:.2f})"


def score_discriminant_tests(
    report_sections: Dict[str, str],
    oracle: Dict[str, Any],
    max_points: int = 15,
) -> Tuple[int, str, int]:
    """Count discriminant tests in Detailed Analysis. Returns (score, note, found_count)."""
    analysis = report_sections.get("Detailed Analysis", "")
    found = 0
    if analysis.strip():
        c1 = len(re.findall(r"(?im)^\s*(?:[-*]\s*)?(?:test|tests?)\s+\d+\b", analysis))
        c2 = len(re.findall(r"(?im)^\s*(?:[-*]\s*)?testable statement\b", normalize_text(analysis)))
        c3 = len(re.findall(r"(?im)^\s*(?:[-*]\s*)?discriminant test\b", normalize_text(analysis)))
        found = max(c1, c2, c3)

    expected_min = oracle.get("expected_min_discriminant_tests") or max(
        2, min(3, len(oracle.get("expected_discriminant_tests", [])) or 2)
    )
    if found >= expected_min:
        return max_points, f"Found {found} discriminant tests (expected ≥{expected_min})", found
    if found == expected_min - 1:
        return max_points - 5, f"Found {found} tests (one short)", found
    return 0, f"Found {found} tests (expected ≥{expected_min})", found


def score_probative_separation(report_text: str, max_points: int = 10) -> Tuple[int, str]:
    """Score presence of all 5 evidence channels in report text. Returns (score, note)."""
    t = normalize_text(report_text)
    groups = {
        "observation": "observation" in t,
        "theory": "theorie" in t or "theory" in t,
        "modeling": "modelisation" in t or "modeling" in t,
        "interpretation": "interpretation" in t,
        "inference": "inference" in t,
    }
    count = sum(1 for v in groups.values() if v)
    if count >= 5:
        return max_points, "Full probative separation"
    if count == 4:
        return 8, "Strong probative separation"
    if count == 3:
        return 6, "Partial probative separation"
    if count == 2:
        return 3, "Weak probative separation"
    return 0, "Probative separation missing"


def score_weakness_taxonomy(
    report_sections: Dict[str, str],
    required_fields: List[str] = DEFAULT_WEAKNESS_FIELDS,
    max_points: int = 10,
) -> Tuple[int, str]:
    """Score weakness micro-syntax completeness. Returns (score, note)."""
    section = report_sections.get("Weakness Taxonomy", "")
    if not section:
        return 0, "Weakness Taxonomy section missing"
    normalized = normalize_text(section)
    missing = [f for f in required_fields if normalize_text(f) not in normalized]
    if not missing:
        return max_points, "Weakness microsyntax present"
    partial = len(required_fields) - len(missing)
    return round(max_points * partial / len(required_fields)), f"Missing weakness fields: {missing}"


def score_fail_closed(
    report_sections: Dict[str, str],
    max_points: int = 10,
) -> Tuple[int, str]:
    """Score fail-closed discipline in Limitations + Summary. Returns (score, note)."""
    limitations = report_sections.get("Limitations", "")
    if not limitations:
        return 0, "Limitations section missing"
    length_ok = len(limitations.strip()) >= 60
    prudence = ["limite", "incertain", "prudent", "insufficient", "absence", "cannot", "fragile",
                "not demonstrated", "heuristic"]
    joined = normalize_text(limitations + "\n" + report_sections.get("Executive Summary", ""))
    hits = sum(1 for p in prudence if p in joined)
    if length_ok and hits >= 2:
        return max_points, "Fail-closed discipline present"
    if length_ok:
        return max_points - 3, "Limitations present but prudence markers sparse"
    return max_points - 6, "Limitations too short"


# ---------------------------------------------------------------------------
# Case JSON validation
# ---------------------------------------------------------------------------

def validate_case_json(case: Dict[str, Any]) -> List[str]:
    """Validate required fields and scoring consistency in a case JSON."""
    issues = []
    required = ["case_id", "title", "claim", "question", "dossier", "oracle", "scoring", "report_contract"]
    issues.extend(f"Missing top-level field: {k}" for k in required if k not in case)

    components = _safe_get(case, ["scoring", "components"], {})
    if components:
        total = sum(int(v) for v in components.values())
        if total != 100:
            issues.append(f"Scoring components sum to {total}, expected 100")

    allowed = _safe_get(case, ["report_contract", "allowed_verdicts"], DEFAULT_ALLOWED_VERDICTS)
    gv = _safe_get(case, ["oracle", "global_verdict"])
    if gv and gv not in allowed:
        issues.append(f"Oracle global verdict not allowed: {gv}")

    for sv in _safe_get(case, ["oracle", "sub_verdicts"], []):
        if sv.get("verdict") not in allowed:
            issues.append(f"Oracle sub-verdict not allowed: {sv.get('verdict')}")
    return issues


# ---------------------------------------------------------------------------
# Master evaluation
# ---------------------------------------------------------------------------

def evaluate(
    case: Dict[str, Any],
    report_text: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """Evaluate a report markdown against a case JSON oracle.

    Args:
        case: Oracle case dict with keys: case_id, oracle, scoring, report_contract, etc.
        report_text: Rendered markdown report.
        strict: If True, return zero score if canonical validation fails.

    Returns: Structured score dict with component scores, total, and differences.
    """
    case_issues = validate_case_json(case)
    rc = case.get("report_contract", {})
    req_sections = rc.get("required_sections", DEFAULT_REQUIRED_SECTIONS)
    allowed_v = rc.get("allowed_verdicts", DEFAULT_ALLOWED_VERDICTS)
    weakness_fields = _safe_get(rc, ["weakness_microsyntax", "fields"], DEFAULT_WEAKNESS_FIELDS)

    raw_sections = extract_sections(report_text)
    canonical_sections: Dict[str, str] = {}
    for k, v in raw_sections.items():
        norm = _normalize_heading(k)
        if norm in req_sections:
            canonical_sections[norm] = v

    missing_sections = [s for s in req_sections if s not in canonical_sections]
    format_issues = list(case_issues)

    verdict = extract_verdict(report_text, allowed_v)
    confidence = extract_confidence(report_text)

    if verdict is None:
        format_issues.append("Missing or unparsable verdict")
    if confidence is None:
        format_issues.append("Missing or unparsable confidence score")

    wt_section = canonical_sections.get("Weakness Taxonomy", "")
    wt_norm = normalize_text(wt_section)
    wt_missing = [f for f in weakness_fields if normalize_text(f) not in wt_norm]
    if wt_missing:
        format_issues.append(f"Weakness microsyntax incomplete: missing {wt_missing}")

    passed = (not missing_sections) and verdict in allowed_v and not wt_missing

    if strict and not passed:
        return {
            "case_id": case.get("case_id"),
            "schema_version": case.get("schema_version", "aegis-case-v1"),
            "canonical_validation": asdict(CanonicalValidation(False, missing_sections, format_issues)),
            "scores": asdict(ComponentScores(0, 0, 0, 0, 0, 0, 0)),
            "bonus": 0, "malus": 0, "total": 0,
            "differences": ["Strict mode: canonical validation failed"],
            "details": {},
        }

    oracle = case["oracle"]
    scoring = case["scoring"]
    comps = scoring["components"]

    s_vs, n_vs, d_vs = score_verdicts(report_text, canonical_sections, oracle, allowed_v, int(comps["global_and_subverdicts"]))
    s_cp, n_cp, uncov = score_critical_points(report_text, oracle.get("critical_points_required", []), int(comps["critical_points_coverage"]))
    s_cf, n_cf = score_confidence(confidence, float(oracle["confidence_score"]), scoring.get("confidence_tolerance", {"full_score_if_abs_diff_lte": 0.5, "light_penalty_if_abs_diff_lte": 1.0, "medium_penalty_if_abs_diff_lte": 2.0}), int(comps["confidence_calibration"]))
    s_dt, n_dt, tests_found = score_discriminant_tests(canonical_sections, oracle, int(comps["discriminant_tests"]))
    s_ps, n_ps = score_probative_separation(report_text, int(comps["probative_separation"]))
    s_wt, n_wt = score_weakness_taxonomy(canonical_sections, weakness_fields, int(comps["weakness_taxonomy"]))
    s_fc, n_fc = score_fail_closed(canonical_sections, int(comps["fail_closed_and_limitations"]))

    # Bonus for extra discriminant tests
    expected_min = oracle.get("expected_min_discriminant_tests") or max(
        2, min(3, len(oracle.get("expected_discriminant_tests", [])) or 2)
    )
    max_bonus = int(_safe_get(scoring, ["bonuses", "original_relevant_test_max"], 0))
    extra = max(0, tests_found - expected_min)
    bonus = min(max_bonus, extra)
    bonus_note = f"Bonus {bonus} for extra tests" if bonus else "No bonus"

    # Fatal error malus
    fatal_policy = case.get("fatal_errors_policy", {"per_error_malus": 5, "max_total_malus": 15})
    fatal_triggered = [err for err in oracle.get("fatal_errors", []) if _semantic_match(err, report_text, threshold=0.28)]
    fatal_malus = min(int(fatal_policy.get("max_total_malus", 15)), int(fatal_policy.get("per_error_malus", 5)) * len(fatal_triggered))

    # Overconfidence malus
    overconf_malus = 0
    if verdict is not None and confidence is not None:
        if verdict != oracle["global_verdict"] and abs(confidence - float(oracle["confidence_score"])) > 1.5:
            overconf_malus = min(int(_safe_get(scoring, ["maluses", "overconfidence_max"], 5)), 5)

    total = s_vs + s_cp + s_cf + s_dt + s_ps + s_wt + s_fc + bonus - fatal_malus - overconf_malus
    total = max(0, min(100 + max_bonus, total))

    differences: List[str] = []
    if verdict != oracle["global_verdict"]:
        differences.append(f"Verdict mismatch: got {verdict}, expected {oracle['global_verdict']}")
    if confidence is not None and abs(confidence - float(oracle["confidence_score"])) > 0.3:
        differences.append(f"Confidence differs by {abs(confidence - float(oracle['confidence_score'])):.1f}")
    if uncov:
        differences.append(f"Uncovered critical points: {uncov}")
    if fatal_triggered:
        differences.append(f"Fatal errors triggered: {fatal_triggered}")
    if missing_sections:
        differences.append(f"Missing sections: {missing_sections}")

    return {
        "case_id": case.get("case_id"),
        "schema_version": case.get("schema_version", "aegis-case-v1"),
        "canonical_validation": asdict(CanonicalValidation(passed, missing_sections, format_issues)),
        "scores": asdict(ComponentScores(
            global_and_subverdicts=s_vs,
            critical_points_coverage=s_cp,
            confidence_calibration=s_cf,
            discriminant_tests=s_dt,
            probative_separation=s_ps,
            weakness_taxonomy=s_wt,
            fail_closed_and_limitations=s_fc,
        )),
        "bonus": bonus,
        "malus": fatal_malus + overconf_malus,
        "total": total,
        "differences": differences,
        "notes": {
            "global_and_subverdicts": n_vs,
            "critical_points_coverage": n_cp,
            "confidence_calibration": n_cf,
            "discriminant_tests": n_dt,
            "probative_separation": n_ps,
            "weakness_taxonomy": n_wt,
            "fail_closed_and_limitations": n_fc,
            "bonus": bonus_note,
        },
        "details": {
            "verdict": verdict,
            "confidence": confidence,
            "expected_verdict": oracle["global_verdict"],
            "expected_confidence": oracle["confidence_score"],
            "uncovered_critical_points": uncov,
            "tests_found": tests_found,
            "fatal_errors_triggered": fatal_triggered,
            "verdict_details": d_vs,
        },
        "claim_ceiling": "LOCAL_LAB_ONLY_NOT_PUBLIC_PROOF",
    }


def score_audit_report(
    report_text: str,
    oracle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a report text with a minimal default oracle (no sub-verdicts required).

    Args:
        report_text: Rendered markdown audit report.
        oracle: Optional oracle dict. If None, uses a default permissive oracle.

    Returns: Score dict with total out of 100 and component breakdown.
    """
    if oracle is None:
        oracle = {
            "global_verdict": extract_verdict(report_text) or "UNCERTAIN",
            "confidence_score": extract_confidence(report_text) or 5.0,
            "sub_verdicts": [],
            "critical_points_required": [],
            "expected_min_discriminant_tests": 1,
            "fatal_errors": [],
        }
    case = {
        "case_id": "default",
        "schema_version": "aegis-case-v1",
        "title": "Standalone report scoring",
        "claim": "",
        "question": "",
        "dossier": "",
        "oracle": oracle,
        "scoring": {
            "components": {
                "global_and_subverdicts": 20,
                "critical_points_coverage": 20,
                "confidence_calibration": 15,
                "discriminant_tests": 15,
                "probative_separation": 10,
                "weakness_taxonomy": 10,
                "fail_closed_and_limitations": 10,
            },
            "confidence_tolerance": {
                "full_score_if_abs_diff_lte": 0.5,
                "light_penalty_if_abs_diff_lte": 1.0,
                "medium_penalty_if_abs_diff_lte": 2.0,
            },
            "bonuses": {"original_relevant_test_max": 0},
            "maluses": {"overconfidence_max": 5},
        },
        "report_contract": {
            "required_sections": DEFAULT_REQUIRED_SECTIONS,
            "allowed_verdicts": DEFAULT_ALLOWED_VERDICTS,
            "weakness_microsyntax": {"fields": DEFAULT_WEAKNESS_FIELDS},
        },
    }
    return evaluate(case, report_text, strict=False)
