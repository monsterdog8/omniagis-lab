"""Coherence metrics: Shannon entropy, mutual information, and replayable coherence passports.

Computes module-level entropy, joint entropy, and mutual information from discretized
numeric observations. Builds and validates coherence passports with SHA-256 hash chains.

All claims: LOCAL_LAB_ONLY — not external proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_PASSPORT_VERSION = "COHERENCE_PASSPORT_v0.1"
_DEFAULT_BINS = 8
_DEFAULT_TOLERANCE = 1e-9

_RAW_PAYLOAD_FIELDS = [
    "cycle_id", "timestamp", "system_version",
    "raw_state_vector", "module_states",
    "entropy_by_module", "total_entropy",
    "mutual_information_matrix", "i_mutual", "h_total",
    "global_coherence", "inter_variance",
    "formula_manifest",
]

_REQUIRED_PASSPORT_FIELDS = _RAW_PAYLOAD_FIELDS + [
    "passport_version", "source_status",
    "raw_payload_hash", "entry_hash", "previous_hash",
    "replay_status", "claim_status", "verdict",
]

_ENTRY_HASH_EXCLUDE = {"entry_hash", "computed_entry_hash", "validation_report"}


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Core information-theory functions (pure stdlib, base-2 bits)
# ---------------------------------------------------------------------------

def shannon_entropy(discrete: Sequence[int]) -> float:
    """Shannon entropy H(X) in bits from a sequence of discrete symbols."""
    n = len(discrete)
    if n == 0:
        return 0.0
    counts = Counter(discrete)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def joint_entropy(x: Sequence[int], y: Sequence[int]) -> float:
    """Joint entropy H(X,Y) in bits."""
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    counts = Counter(zip(x[:n], y[:n]))
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    """Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y), clipped to [0, ∞)."""
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    return max(0.0, float(shannon_entropy(x[:n]) + shannon_entropy(y[:n]) - joint_entropy(x[:n], y[:n])))


def global_coherence(i_mutual: float, h_total: float) -> float:
    """C(S) = I_mutual(S) / H_total(S); returns 0 if h_total == 0."""
    return float(i_mutual / h_total) if h_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def _make_edges(values: Sequence[float], bins: int) -> List[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [lo, hi]
    w = (hi - lo) / bins
    return [lo + w * i for i in range(bins + 1)]


def digitize(values: Sequence[float], bins: int = _DEFAULT_BINS) -> List[int]:
    """Bin continuous values into discrete integer indices in [0, bins-1]."""
    edges = _make_edges(values, bins)
    if len(edges) == 2 and math.isclose(edges[0], edges[1]):
        return [0] * len(values)
    out: List[int] = []
    for v in values:
        if v >= edges[-1]:
            out.append(len(edges) - 2)
            continue
        idx = 0
        while idx < len(edges) - 1 and not (edges[idx] <= v < edges[idx + 1]):
            idx += 1
        out.append(max(0, min(idx, len(edges) - 2)))
    return out


# ---------------------------------------------------------------------------
# Metric field computation from module observations
# ---------------------------------------------------------------------------

def compute_metric_fields(
    observations: Dict[str, List[float]],
    bins: int = _DEFAULT_BINS,
) -> Dict[str, Any]:
    """Compute entropy, mutual information matrix, and coherence from module observations.

    Args:
        observations: {module_id: [float, ...]} — each list has ≥2 numeric samples.
        bins: Number of discretization bins.

    Returns dict with keys: module_ids, entropy_by_module, h_total, mutual_information_matrix,
        i_mutual, global_coherence, inter_variance, raw_state_vector, module_states.
    """
    module_ids = list(observations.keys())
    discrete = {mid: digitize(vals, bins=bins) for mid, vals in observations.items()}
    entropy_by_module = {mid: shannon_entropy(discrete[mid]) for mid in module_ids}
    h_total = float(sum(entropy_by_module.values()))

    matrix: List[List[float]] = []
    for a in module_ids:
        row: List[float] = []
        for b in module_ids:
            row.append(0.0 if a == b else mutual_information(discrete[a], discrete[b]))
        matrix.append(row)

    i_mutual = sum(
        matrix[i][j]
        for i in range(len(module_ids))
        for j in range(i + 1, len(module_ids))
    )

    latest = [observations[mid][-1] for mid in module_ids]
    mean_v = sum(latest) / len(latest)
    inter_variance = sum((v - mean_v) ** 2 for v in latest) / len(latest)

    return {
        "module_ids": module_ids,
        "raw_state_vector": latest,
        "module_states": [
            {
                "module_id": mid,
                "latest_state": observations[mid][-1],
                "sample_count": len(observations[mid]),
                "sample_min": min(observations[mid]),
                "sample_max": max(observations[mid]),
            }
            for mid in module_ids
        ],
        "entropy_by_module": entropy_by_module,
        "total_entropy": h_total,
        "h_total": h_total,
        "mutual_information_matrix": matrix,
        "i_mutual": float(i_mutual),
        "global_coherence": global_coherence(i_mutual, h_total),
        "inter_variance": float(inter_variance),
    }


# ---------------------------------------------------------------------------
# Observation normalization and validation
# ---------------------------------------------------------------------------

def _is_finite_float(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def normalize_observations(
    module_observations: Any,
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Validate and normalize module_observations into float lists.

    Returns (normalized_dict, blockers, warnings).
    """
    blockers: List[str] = []
    warnings: List[str] = []

    if not isinstance(module_observations, Mapping) or not module_observations:
        return {}, ["BLOCKED_NO_MODULE_OBSERVATIONS"], warnings

    normalized: Dict[str, List[float]] = {}
    for mid, samples in sorted(module_observations.items(), key=lambda kv: str(kv[0])):
        if not isinstance(samples, (list, tuple)) or isinstance(samples, (str, bytes)):
            blockers.append(f"BLOCKED_TOO_FEW_SAMPLES:{mid}")
            continue
        vals: List[float] = []
        for s in samples:
            if not _is_finite_float(s):
                blockers.append(f"BLOCKED_NON_NUMERIC_SAMPLE:{mid}")
                vals = []
                break
            vals.append(float(s))
        if len(vals) < 2:
            blockers.append(f"BLOCKED_TOO_FEW_SAMPLES:{mid}")
            continue
        if len(vals) < 16:
            warnings.append(f"SMALL_SAMPLE:{mid}:n={len(vals)}")
        normalized[str(mid)] = vals

    return normalized, blockers, warnings


# ---------------------------------------------------------------------------
# Passport construction and hash sealing
# ---------------------------------------------------------------------------

def _raw_payload(passport: Dict[str, Any]) -> Dict[str, Any]:
    return {k: passport.get(k) for k in _RAW_PAYLOAD_FIELDS}


def _entry_payload(passport: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in passport.items() if k not in _ENTRY_HASH_EXCLUDE}


def build_coherence_passport(
    cycle_id: Any,
    module_observations: Mapping[str, Any],
    *,
    system_version: str = "LOCAL_CAPTURE",
    previous_hash: Optional[str] = None,
    bins: int = _DEFAULT_BINS,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a coherence passport from raw module observations.

    If observations are invalid, returns a BLOCKED fail-closed passport.
    Hashes are unsealed; call seal_passport_hashes() before replay.
    """
    observations, blockers, warnings = normalize_observations(module_observations)
    warnings.append("UNCALIBRATED_BINS_LOCAL_PROXY")

    base: Dict[str, Any] = {
        "passport_version": _PASSPORT_VERSION,
        "cycle_id": cycle_id,
        "timestamp": _utc_now(),
        "system_version": system_version,
        "source_status": "LOCAL_CAPTURE" if not blockers else "CAPTURE_BLOCKED",
        "lab_only": True,
        "production_status": "LOCKED",
        "public_proof": "NOT_GRANTED",
        "formula_manifest": {
            "coherence_formula": "C(S)=I_mutual(S)/H_total(S)",
            "i_mutual_policy": "upper_triangle_sum_empirical_discrete_mi",
            "h_total_policy": "sum_module_shannon_entropy_discretized",
            "entropy_base": 2,
            "bins": bins,
            "tolerance": _DEFAULT_TOLERANCE,
            "formula_verified": False,
            "calibration_status": "UNCALIBRATED_LOCAL_PROXY",
        },
        "raw_payload_hash": None,
        "entry_hash": None,
        "previous_hash": previous_hash,
        "replay_script_hash": None,
        "replay_status": "NOT_RUN",
        "claim_status": "LOCKED",
        "verdict": "UNVALIDATED_CAPTURE_PASSPORT",
        "blockers": blockers,
        "warnings": warnings,
    }
    if meta:
        base.update({k: v for k, v in meta.items() if k not in base})

    if blockers:
        base.update({
            "raw_state_vector": [],
            "module_states": [],
            "entropy_by_module": {},
            "total_entropy": None,
            "mutual_information_matrix": [],
            "i_mutual": None,
            "h_total": None,
            "global_coherence": None,
            "inter_variance": None,
            "verdict": "CAPTURE_BLOCKED_FAIL_CLOSED",
        })
        return base

    fields = compute_metric_fields(observations, bins=bins)
    base.update({
        "raw_state_vector": fields["raw_state_vector"],
        "module_states": fields["module_states"],
        "entropy_by_module": fields["entropy_by_module"],
        "total_entropy": fields["total_entropy"],
        "h_total": fields["h_total"],
        "mutual_information_matrix": fields["mutual_information_matrix"],
        "i_mutual": fields["i_mutual"],
        "global_coherence": fields["global_coherence"],
        "inter_variance": fields["inter_variance"],
    })
    return base


def seal_passport_hashes(passport: Dict[str, Any]) -> Dict[str, Any]:
    """Fill raw_payload_hash and entry_hash in a passport copy. Does not invent missing raw data."""
    p = copy.deepcopy(passport)
    p["raw_payload_hash"] = _sha256_json(_raw_payload(p))
    p["entry_hash"] = _sha256_json(_entry_payload(p))
    return p


# ---------------------------------------------------------------------------
# Passport validation and replay
# ---------------------------------------------------------------------------

def validate_coherence_passport(
    passport: Dict[str, Any],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """Validate a sealed coherence passport. Returns a report dict with verdict.

    Checks: required fields, formula consistency (i_mutual/h_total == global_coherence),
    raw_payload_hash, entry_hash.
    """
    p = copy.deepcopy(passport)
    errors: List[str] = []
    warnings: List[str] = []

    missing = [f for f in _REQUIRED_PASSPORT_FIELDS if f not in p]
    if missing:
        errors.extend(f"MISSING_FIELD:{f}" for f in missing)

    source = p.get("source_status", "")
    if source in {"INTERFACE_SIGNAL_ONLY", "DASHBOARD_ONLY", "WORLD_RENDERER_ONLY"}:
        errors.append("WORLD_NOT_PROOF")

    rsv = p.get("raw_state_vector")
    ms = p.get("module_states")
    if not (isinstance(rsv, list) and rsv) or not (isinstance(ms, list) and ms):
        errors.append("BLOCKED_NO_RAW")

    ebm = p.get("entropy_by_module")
    if not (isinstance(ebm, dict) and ebm):
        errors.append("BLOCKED_NO_ENTROPY_BY_MODULE")

    mi_mat = p.get("mutual_information_matrix")
    if not (isinstance(mi_mat, list) and mi_mat):
        errors.append("BLOCKED_NO_MI_MATRIX")

    fm = p.get("formula_manifest")
    if not (isinstance(fm, dict) and fm.get("coherence_formula")):
        errors.append("BLOCKED_NO_FORMULA")

    i_mutual = p.get("i_mutual")
    h_total = p.get("h_total")
    gc = p.get("global_coherence")

    computed_gc: Optional[float] = None
    formula_verified = False
    if (
        isinstance(i_mutual, (int, float)) and math.isfinite(float(i_mutual)) and float(i_mutual) >= 0
        and isinstance(h_total, (int, float)) and math.isfinite(float(h_total)) and float(h_total) > 0
    ):
        computed_gc = float(i_mutual) / float(h_total)
        if isinstance(gc, (int, float)) and math.isfinite(float(gc)):
            if abs(computed_gc - float(gc)) <= tolerance:
                formula_verified = True
            else:
                errors.append("BLOCKED_FORMULA_MISMATCH")
        else:
            errors.append("BLOCKED_NO_GLOBAL_COHERENCE")
    else:
        if not (isinstance(i_mutual, (int, float)) and math.isfinite(float(i_mutual if i_mutual is not None else float("nan")))):
            errors.append("BLOCKED_NO_I_MUTUAL")
        if not (isinstance(h_total, (int, float)) and math.isfinite(float(h_total if h_total is not None else float("nan"))) and float(h_total if h_total is not None else 0) > 0):
            errors.append("BLOCKED_NO_H_TOTAL")

    computed_raw_hash = _sha256_json(_raw_payload(p))
    if p.get("raw_payload_hash") != computed_raw_hash:
        errors.append("BLOCKED_RAW_HASH_MISMATCH")

    computed_entry_hash = _sha256_json(_entry_payload(p))
    if p.get("entry_hash") != computed_entry_hash:
        errors.append("BLOCKED_ENTRY_HASH_MISMATCH")

    if p.get("previous_hash") is None:
        warnings.append("PREVIOUS_HASH_NULL_GENESIS_OR_UNCHAINED")

    replay_status = "FAIL"
    claim_status = "LOCKED"
    if errors:
        verdict = "INTERFACE_SIGNAL_ONLY_FAIL_CLOSED" if "WORLD_NOT_PROOF" in errors or "BLOCKED_NO_RAW" in errors else "BLOCKED_FAIL_CLOSED"
    else:
        replay_status = "PASS"
        claim_status = "BOUND_LOCAL_COHERENCE_CLAIM_ALLOWED"
        verdict = "PASS_REPLAYABLE_LOCAL_COHERENCE_PASSPORT"

    return {
        "passport_version": p.get("passport_version"),
        "cycle_id": p.get("cycle_id"),
        "timestamp": _utc_now(),
        "lab_only": True,
        "production_status": "LOCKED",
        "public_proof": "NOT_GRANTED",
        "errors": errors,
        "warnings": warnings,
        "formula_verified": formula_verified,
        "computed_global_coherence": computed_gc,
        "computed_raw_payload_hash": computed_raw_hash,
        "computed_entry_hash": computed_entry_hash,
        "replay_status": replay_status,
        "claim_status": claim_status,
        "verdict": verdict,
        "claim_ceiling": "LOCAL_LAB_ONLY_NOT_PUBLIC_PROOF",
    }
