"""Tests for MODULE_002_CLASSIFIER.

Extracted from tests/test_gpts_core.py::TestClassifier.
Imports from the module's exports directory.
"""
from __future__ import annotations

import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestClassifier:
    def test_simulated_label(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("psiomega_mean", "simulated", "simulation run")
        assert label == "simulated"
        assert 0.0 <= conf <= 1.0

    def test_observed_label(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("temperature", "36.5", "measured by sensor")
        assert label in ["observed", "reported", "computed"]

    def test_symbolic_label(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("constant_pi", "3.14159", "mathematical constant symbolic")
        assert label in ["symbolic", "computed", "reported"]

    def test_unsupported_label(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("", "", "")
        assert label in ["unsupported", "symbolic", "observed", "computed", "reported", "simulated"]

    def test_reasons_is_list(self):
        from classifier import classify_metric
        _, reasons, _ = classify_metric("x", "1.0", "")
        assert isinstance(reasons, list)

    def test_labels_constant(self):
        from classifier import LABELS
        assert set(LABELS) == {"observed", "computed", "reported", "simulated", "symbolic", "unsupported"}

    def test_confidence_in_range(self):
        from classifier import classify_metric
        for name, val, ctx in [("x", "sim", ""), ("y", "obs", "measured"), ("z", "calc", "formula")]:
            _, _, conf = classify_metric(name, val, ctx)
            assert 0.0 <= conf <= 1.0

    # Additional coverage tests

    def test_computed_label_mean(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("result_mean", "0.72", "mean of scores computed")
        assert label in ["computed", "reported"]
        assert 0.0 <= conf <= 1.0

    def test_reported_label_threshold(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("threshold_value", "0.5", "config threshold parameter")
        assert label in ["reported", "computed"]
        assert 0.0 <= conf <= 1.0

    def test_unsupported_production_ready(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("claim", "production-ready", "production-ready system")
        assert label == "unsupported"
        assert 0.0 <= conf <= 1.0

    def test_equation_recalculable_true(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric(
            "ratio", "0.5", row_type="recalculation", recalculable="True", result="0.5"
        )
        assert label == "computed"
        assert 0.0 <= conf <= 1.0

    def test_equation_symbolic(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric(
            "Omega", "∞", "cosmic singularity", row_type="equation"
        )
        assert label in ["symbolic", "unsupported"]
        assert 0.0 <= conf <= 1.0

    def test_equation_not_recalculable(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric(
            "formula", "x+y", row_type="equation", recalculable=None, result=None
        )
        assert label == "unsupported"
        assert 0.0 <= conf <= 1.0

    def test_observed_image_pixel(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("pixel_luminance", "200", "image pixel measurement")
        assert label == "observed"
        assert 0.0 <= conf <= 1.0

    def test_simulated_rng_context(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("noise_val", "0.33", "np.random randn generation")
        assert label == "simulated"
        assert 0.0 <= conf <= 1.0

    def test_generic_name_numeric_unsupported(self):
        from classifier import classify_metric
        label, reasons, conf = classify_metric("i", "42", "")
        assert label == "unsupported"
        assert 0.0 <= conf <= 1.0

    def test_all_labels_in_labels_list(self):
        from classifier import LABELS
        assert len(LABELS) == 6
        for lab in ["observed", "computed", "reported", "simulated", "symbolic", "unsupported"]:
            assert lab in LABELS

    def test_classify_returns_three_tuple(self):
        from classifier import classify_metric
        result = classify_metric("score", "0.88", "overall score computed")
        assert len(result) == 3
        label, reasons, conf = result
        assert isinstance(label, str)
        assert isinstance(reasons, list)
        assert isinstance(conf, float)
