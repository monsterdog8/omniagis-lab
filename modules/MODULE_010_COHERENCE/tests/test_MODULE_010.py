"""Tests for MODULE_010_COHERENCE — extracted from tests/test_gpts_core.py::TestCoherence."""
from __future__ import annotations

import random

import pytest

from gpts_core.coherence import (
    shannon_entropy,
    mutual_information,
    global_coherence,
    compute_metric_fields,
    build_coherence_passport,
    validate_coherence_passport,
)


class TestCoherence:
    # shannon_entropy takes Sequence[int] (discrete symbol indices), not probabilities

    def test_shannon_entropy_uniform_binary(self):
        # Two equally likely symbols: H = 1.0 bit
        symbols = [0, 1] * 50
        h = shannon_entropy(symbols)
        assert abs(h - 1.0) < 0.01

    def test_shannon_entropy_certain(self):
        # One symbol: H = 0.0 bits
        h = shannon_entropy([0] * 10)
        assert h == 0.0

    def test_shannon_entropy_max(self):
        # 4 equally likely symbols: H = 2.0 bits
        symbols = [0, 1, 2, 3] * 10
        h = shannon_entropy(symbols)
        assert abs(h - 2.0) < 0.01

    def test_mutual_information_independent(self):
        # Two independent uniform sequences
        rng = random.Random(42)
        x = [rng.randint(0, 3) for _ in range(200)]
        y = [rng.randint(0, 3) for _ in range(200)]
        mi = mutual_information(x, y)
        assert mi >= 0.0

    def test_mutual_information_dependent(self):
        # Fully correlated: I(X;Y) > 0
        x = [0] * 50 + [1] * 50
        y = [0] * 50 + [1] * 50
        mi = mutual_information(x, y)
        assert mi > 0

    def test_global_coherence_range(self):
        c = global_coherence(0.6, 1.2)
        assert 0.0 <= c <= 1.0

    def test_global_coherence_zero_total(self):
        c = global_coherence(0.0, 0.0)
        assert c == 0.0

    def test_compute_metric_fields(self):
        obs = {"mod_A": [0.1 * i for i in range(20)],
               "mod_B": [0.05 * i for i in range(20)]}
        fields = compute_metric_fields(obs)
        assert "i_mutual" in fields
        assert "h_total" in fields
        assert "global_coherence" in fields
        assert 0.0 <= fields["global_coherence"] <= 1.0

    def test_build_coherence_passport(self):
        obs = {"mod_A": [float(i) for i in range(20)],
               "mod_B": [float(i % 5) for i in range(20)]}
        passport = build_coherence_passport("C88", obs, meta={"cycle": "C88"})
        assert isinstance(passport, dict)

    def test_validate_coherence_passport_returns_dict(self):
        obs = {"mod_A": [float(i) for i in range(20)],
               "mod_B": [float(i % 5) for i in range(20)]}
        passport = build_coherence_passport("C88", obs)
        result = validate_coherence_passport(passport)
        assert isinstance(result, dict)
        assert "verdict" in result or "errors" in result

    def test_validate_coherence_passport_empty(self):
        result = validate_coherence_passport({})
        assert isinstance(result, dict)
        errors = result.get("errors", [])
        assert len(errors) > 0 or result.get("verdict") in ("BLOCKED", "FAIL", "INVALID")
