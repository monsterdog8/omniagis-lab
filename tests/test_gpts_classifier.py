"""Tests for gpts_core.classifier — metric text classification."""
from __future__ import annotations

import pytest

from gpts_core.classifier import classify_metric, LABELS


# ---------------------------------------------------------------------------
# Label set
# ---------------------------------------------------------------------------

class TestLabelSet:
    def test_six_labels(self):
        assert len(LABELS) == 6

    def test_expected_labels(self):
        assert set(LABELS) == {"observed", "computed", "reported", "simulated", "symbolic", "unsupported"}


# ---------------------------------------------------------------------------
# Simulated
# ---------------------------------------------------------------------------

class TestSimulated:
    def test_random_pattern(self):
        label, reasons, conf = classify_metric("x", "42", context="np.random.uniform()")
        assert label == "simulated"
        assert conf == pytest.approx(0.86)

    def test_simulation_keyword(self):
        label, _, _ = classify_metric("sim_score", "1.0", context="simulation run")
        assert label == "simulated"

    def test_synthetic_keyword(self):
        label, _, _ = classify_metric("metric", "3", context="synthetic data")
        assert label == "simulated"

    def test_reason_present(self):
        _, reasons, _ = classify_metric("m", "1", context="random")
        assert "simulation_or_random_pattern" in reasons


# ---------------------------------------------------------------------------
# Unsupported (direct match)
# ---------------------------------------------------------------------------

class TestUnsupported:
    def test_global_superiority(self):
        label, _, _ = classify_metric("m", "1", context="global superiority claim")
        assert label == "unsupported"

    def test_production_ready(self):
        label, _, _ = classify_metric("status", "ok", context="production-ready system")
        assert label == "unsupported"


# ---------------------------------------------------------------------------
# Observed
# ---------------------------------------------------------------------------

class TestObserved:
    def test_observed_keyword(self):
        label, _, _ = classify_metric("luminance", "128", context="measured from image")
        assert label == "observed"

    def test_photo_context(self):
        label, _, _ = classify_metric("edge_count", "42", context="photo measurement")
        assert label == "observed"

    def test_reason_present(self):
        _, reasons, _ = classify_metric("distance_cm", "5.2", context="measured")
        assert any("observational" in r for r in reasons)


# ---------------------------------------------------------------------------
# Symbolic
# ---------------------------------------------------------------------------

class TestSymbolic:
    def test_symbolic_keyword(self):
        label, _, _ = classify_metric("psi", "Ω", context="consciousness metric")
        assert label == "symbolic"

    def test_unicode_symbol(self):
        label, _, _ = classify_metric("field", "∞", context="")
        assert label == "symbolic"


# ---------------------------------------------------------------------------
# Computed
# ---------------------------------------------------------------------------

class TestComputed:
    def test_mean_keyword(self):
        label, _, _ = classify_metric("accuracy_mean", "0.85", context="mean across runs")
        assert label == "computed"

    def test_sha256_is_computed(self):
        label, _, _ = classify_metric("sha256_checksum", "abc123", context="sha256 computed")
        assert label == "computed"

    def test_generic_variable_name_goes_to_unsupported(self):
        label, _, _ = classify_metric("x", "42", context="mean value")
        assert label == "unsupported"
        assert any("generic" in r for r in classify_metric("x", "42", context="mean value")[1])


# ---------------------------------------------------------------------------
# Reported
# ---------------------------------------------------------------------------

class TestReported:
    def test_threshold_keyword(self):
        label, _, _ = classify_metric("safety_threshold", "0.9", context="config")
        assert label == "reported"

    def test_verdict_keyword(self):
        label, _, _ = classify_metric("verdict", "PASS", context="status reported")
        assert label == "reported"


# ---------------------------------------------------------------------------
# Equation / recalculation row_type
# ---------------------------------------------------------------------------

class TestEquationRowType:
    def test_recalculable_computed(self):
        label, reasons, conf = classify_metric(
            "sum_score", "42.0", row_type="recalculation",
            recalculable="True", result="42.0"
        )
        assert label == "computed"
        assert conf == pytest.approx(0.78)

    def test_equation_not_recalculable(self):
        label, _, _ = classify_metric("score", "x", row_type="equation",
                                       recalculable="False", result=None)
        assert label == "unsupported"

    def test_equation_symbolic(self):
        label, _, _ = classify_metric("psi_Ω", "∞", row_type="equation",
                                       recalculable="False", result=None)
        assert label == "symbolic"


# ---------------------------------------------------------------------------
# Numeric fallback
# ---------------------------------------------------------------------------

class TestNumericFallback:
    def test_numeric_non_generic_is_reported(self):
        label, _, _ = classify_metric("stability_score", "0.75")
        assert label == "reported"

    def test_numeric_generic_single_char_is_unsupported(self):
        label, _, _ = classify_metric("i", "1")
        assert label == "unsupported"

    def test_non_numeric_no_context_is_unsupported(self):
        label, _, conf = classify_metric("something", "NOT_A_NUMBER")
        assert label == "unsupported"
        assert conf == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Confidence ranges
# ---------------------------------------------------------------------------

class TestConfidenceValues:
    @pytest.mark.parametrize("context,expected_label,expected_conf", [
        ("np.random.randn()", "simulated", 0.86),
        ("global superiority", "unsupported", 0.82),
        ("measured luminance", "observed", 0.74),
        ("consciousness metric", "symbolic", 0.70),
    ])
    def test_confidence(self, context, expected_label, expected_conf):
        label, _, conf = classify_metric("m", "1", context=context)
        assert label == expected_label
        assert conf == pytest.approx(expected_conf)
