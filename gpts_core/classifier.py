"""Metric text classification.

Classifies extracted metric rows into audit categories using regex heuristics.
Labels: observed / computed / reported / simulated / symbolic / unsupported

This is a heuristic classifier — labels require manual review before use as evidence.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

LABELS = ["observed", "computed", "reported", "simulated", "symbolic", "unsupported"]

_SIM = re.compile(
    r"\b(random|uniform\(|randn|randint|rng|np\.random|simulation|simulated|"
    r"synthetic|fake|mock|dummy|benchmark_rows|fractal_iterations)\b", re.I
)
_OBS = re.compile(
    r"\b(observed|measured|measurement|luminance|rgb|histogram|mask_dark|mask_bright|"
    r"distance|cm\b|photo|image|pixel|edge|saturation|dark_ratio|bright_ratio)\b", re.I
)
_COMP = re.compile(
    r"\b(mean|median|sum|count|ratio|round\(|sqrt|log|exp|abs\(|min\(|max\(|std|"
    r"variance|recalculation|computed|calculated|score\s*=|sha256|hash)\b", re.I
)
_REPORT = re.compile(
    r"\b(threshold|safe_hold|status|verdict|version|files_scanned|readable_files|"
    r"claim_flags|metric_rows|overall_score|fail_rate|rows|generated|reported|"
    r"config|constant|parameter)\b", re.I
)
_SYMBOL = re.compile(
    r"[∞ΩΨÆ∑∆∇⟁𓂀]|\b(consciousness|fractal|aether|quantum|omega|singularity|"
    r"entity|living|sentience|sacred|cosmic)\b", re.I
)
_UNSUPPORTED = re.compile(
    r"\b(global superiority|domination|world.?class|leaderboard|clinical|biomedical|"
    r"diagnosis|therapy|production.ready|autonomous consciousness|sentient)\b", re.I
)
_GENERIC = {"axis", "i", "j", "k", "x", "y", "z", "n", "m", "row", "col", "idx", "index", "seed", "_"}


def _is_numeric(value: str) -> bool:
    try:
        return math.isfinite(float(str(value).strip()))
    except Exception:
        return False


def classify_metric(
    metric_name: str,
    value_raw: str,
    context: str = "",
    path: str = "",
    row_type: str = "metric",
    recalculable: Optional[str] = None,
    result: Optional[str] = None,
) -> Tuple[str, List[str], float]:
    """Classify a metric row.

    Returns (label, reasons, confidence) where confidence is in [0, 1].
    label is one of: observed / computed / reported / simulated / symbolic / unsupported
    """
    joined = " ".join(filter(None, [metric_name, value_raw, context, path, row_type,
                                     recalculable or "", result or ""]))
    name = (metric_name or "").strip().lower()
    reasons: List[str] = []

    if _SIM.search(joined):
        reasons.append("simulation_or_random_pattern")
        return "simulated", reasons, 0.86

    if _UNSUPPORTED.search(joined):
        reasons.append("unsupported_high_claim_pattern")
        return "unsupported", reasons, 0.82

    if row_type in {"equation", "recalculation"}:
        if recalculable == "True" and _is_numeric(result or ""):
            reasons.append("safe_arithmetic_recalculation")
            return "computed", reasons, 0.78
        if _SYMBOL.search(joined):
            reasons.append("symbolic_equation")
            return "symbolic", reasons, 0.72
        reasons.append("equation_not_recalculable")
        return "unsupported", reasons, 0.66

    if _OBS.search(joined):
        reasons.append("observational_or_image_measurement")
        return "observed", reasons, 0.74

    if _SYMBOL.search(joined):
        reasons.append("symbolic_or_metaphoric_context")
        return "symbolic", reasons, 0.70

    if _COMP.search(joined):
        if name in _GENERIC:
            reasons.append("generic_variable_name_with_computation_context")
            return "unsupported", reasons, 0.62
        reasons.append("computed_metric_pattern")
        return "computed", reasons, 0.68

    if _REPORT.search(joined):
        reasons.append("reported_config_or_status_pattern")
        return "reported", reasons, 0.66

    if _is_numeric(value_raw):
        if name in _GENERIC or len(name) <= 1:
            reasons.append("numeric_generic_variable")
            return "unsupported", reasons, 0.58
        reasons.append("numeric_value_without_raw_observation")
        return "reported", reasons, 0.55

    reasons.append("insufficient_parseable_context")
    return "unsupported", reasons, 0.55
