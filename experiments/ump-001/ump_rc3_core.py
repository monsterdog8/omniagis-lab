#!/usr/bin/env python3
"""UMP-001 GTAB_RC3 child core.

Patch-forward child of Stage0. This module does not authorize a real GTAB trial.
It implements only deterministic engineering primitives:
- ordinary RC prediction (C2 support primitive),
- proper-score comparison versus a declared null,
- commit/reveal byte integrity for machine-readable predictions,
- unbiased 3-way assignment from a 256-bit digest via rejection sampling.

No function in this file establishes physical RNG independence, future leakage closure,
C4 evidence, consciousness, retrocausality, or new physics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

PARENT_STAGE0_CAPSULE_SHA256 = (
    "3d000c5ec1ec27d1fde68fa528ac25e9d5c34b698e3cc7f684682cf180f3057c"
)
COMMIT_DOMAIN = b"UMP1-COMMIT-v1"
THREE_WAY_REJECT = (1 << 256) - 1


def canonical_json_bytes(obj: object) -> bytes:
    """Deterministic engineering serialization.

    Stage0's frozen manifest states its JSON stayed inside a JCS-safe primitive subset.
    Confirmatory commitments remain contractually RFC 8785 JCS; this helper is a local
    engineering implementation for the child selftests and is not a substitute for a
    separately validated RFC 8785 implementation if richer JSON is later introduced.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rc_voltage(*, r_ohm: float, c_farad: float, v0: float, vin: float, delta_s: float) -> float:
    values = (r_ohm, c_farad, v0, vin, delta_s)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("all inputs must be finite")
    if r_ohm <= 0 or c_farad <= 0:
        raise ValueError("R and C must be > 0")
    if delta_s < 0:
        raise ValueError("delta_s must be >= 0")
    tau = r_ohm * c_farad
    return vin + (v0 - vin) * math.exp(-delta_s / tau)


def validate_probabilities(probs: Mapping[str, float]) -> tuple[str, ...]:
    if not isinstance(probs, Mapping) or not probs:
        raise ValueError("probabilities must be a non-empty mapping")
    keys = tuple(sorted(str(k) for k in probs))
    vals = [float(probs[k]) for k in keys]
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals):
        raise ValueError("probabilities must be finite and within [0,1]")
    if abs(sum(vals) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to 1")
    return keys


def negative_brier(probs: Mapping[str, float], outcome: str) -> float:
    keys = validate_probabilities(probs)
    if outcome not in probs:
        raise ValueError("outcome absent from prediction classes")
    return -sum((float(probs[k]) - (1.0 if k == outcome else 0.0)) ** 2 for k in keys)


def log_score(probs: Mapping[str, float], outcome: str, epsilon: float = 1e-12) -> float:
    validate_probabilities(probs)
    if outcome not in probs:
        raise ValueError("outcome absent from prediction classes")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")
    return math.log(max(float(probs[outcome]), epsilon))


def proper_score_skill(
    model_probs: Mapping[str, float],
    null_probs: Mapping[str, float],
    outcome: str,
) -> dict[str, float]:
    model_keys = validate_probabilities(model_probs)
    null_keys = validate_probabilities(null_probs)
    if model_keys != null_keys:
        raise ValueError("model and null class sets must match")
    mb = negative_brier(model_probs, outcome)
    nb = negative_brier(null_probs, outcome)
    ml = log_score(model_probs, outcome)
    nl = log_score(null_probs, outcome)
    return {
        "negative_brier_model": mb,
        "negative_brier_null": nb,
        "brier_skill": mb - nb,
        "log_score_model": ml,
        "log_score_null": nl,
        "log_skill": ml - nl,
    }


def uniform_null(classes: Sequence[str]) -> dict[str, float]:
    classes = tuple(str(x) for x in classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain >=2 unique labels")
    p = 1.0 / len(classes)
    return {k: p for k in classes}


def commit_prediction(payload: object, nonce: bytes) -> str:
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) < 32:
        raise ValueError("nonce must be at least 32 bytes")
    blob = COMMIT_DOMAIN + b"\x00" + canonical_json_bytes(payload) + b"\x00" + bytes(nonce)
    return sha256_hex(blob)


def verify_commitment(payload: object, nonce: bytes, expected_hex: str) -> bool:
    return commit_prediction(payload, nonce) == str(expected_hex).lower()


def assignment_from_digest_three_way(digest32: bytes) -> int | None:
    """Return 0/1/2, or None for the single rejected uint256 value.

    Since 2**256 mod 3 == 1, rejecting 2**256-1 makes the accepted domain
    exactly divisible by 3 and removes modulo bias.
    """
    if not isinstance(digest32, (bytes, bytearray)) or len(digest32) != 32:
        raise ValueError("digest32 must contain exactly 32 bytes")
    x = int.from_bytes(digest32, "big")
    if x == THREE_WAY_REJECT:
        return None
    return x % 3


def postcommit_assignment_digest(
    *,
    local_entropy: bytes,
    future_external_value: bytes,
    trial_id: str,
    commit_hash_hex: str,
) -> bytes:
    if len(local_entropy) < 32:
        raise ValueError("local_entropy must contain >=32 bytes")
    if not future_external_value:
        raise ValueError("future_external_value is required")
    if not trial_id:
        raise ValueError("trial_id is required")
    try:
        commit_raw = bytes.fromhex(commit_hash_hex)
    except ValueError as exc:
        raise ValueError("commit_hash_hex must be hexadecimal") from exc
    if len(commit_raw) != 32:
        raise ValueError("commit_hash_hex must encode 32 bytes")
    h = hashlib.sha256()
    h.update(b"UMP1-ASSIGN-v1\x00")
    h.update(local_entropy)
    h.update(b"\x00")
    h.update(future_external_value)
    h.update(b"\x00")
    h.update(trial_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(commit_raw)
    return h.digest()


@dataclass(frozen=True)
class GateState:
    gtab_real_bound: bool = False
    observation_device_bound: bool = False
    precision_bound: bool = False
    sampling_bound: bool = False
    d_primary_bound: bool = False
    delta_primary_bound: bool = False
    physical_rng_independence_proven: bool = False

    @property
    def engineering_pilot_ready(self) -> bool:
        return all(
            (
                self.gtab_real_bound,
                self.observation_device_bound,
                self.precision_bound,
                self.sampling_bound,
                self.d_primary_bound,
                self.delta_primary_bound,
            )
        )

    @property
    def c4_confirmatory_ready(self) -> bool:
        return self.engineering_pilot_ready and self.physical_rng_independence_proven


if __name__ == "__main__":
    p = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert abs(negative_brier(p, "A") + 2 / 3) < 1e-12
    assert assignment_from_digest_three_way(b"\xff" * 32) is None
    print("UMP_001_RC3_CORE_SELFTEST_PASS")
