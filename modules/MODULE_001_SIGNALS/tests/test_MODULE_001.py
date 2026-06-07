"""Tests for MODULE_001_SIGNALS.

Extracted from tests/test_gpts_core.py::TestSignals.
Imports from the module's exports directory.
"""
from __future__ import annotations

import math
import sys
import pathlib

# Allow imports from the exports directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "exports"))

import pytest


class TestSignals:
    def _sine_wave(self, n=256, freq=8.0, sr=256.0):
        import math
        return [math.sin(2 * math.pi * freq * i / sr) for i in range(n)]

    def test_entropy_norm_range(self):
        from signals import entropy_norm
        x = self._sine_wave()
        h = entropy_norm(x)
        assert 0.0 <= h <= 1.0

    def test_entropy_uniform_higher_than_constant(self):
        from signals import entropy_norm
        uniform = list(range(256))
        constant = [0.5] * 256
        assert entropy_norm(uniform) > entropy_norm(constant)

    def test_autocorr_returns_two_floats(self):
        from signals import autocorr
        peak, mean_ac = autocorr(self._sine_wave())
        assert isinstance(peak, float)
        assert isinstance(mean_ac, float)

    def test_structure_score_range(self):
        from signals import structure_score
        row = {
            "entropy_norm": 0.6,
            "autocorr_max": 0.5,
            "spectral_concentration": 0.07,
            "ar2_r2": 0.4,
            "motif_top_share": 0.04,
            "compression_ratio": 0.5,
            "peak_freq_cv": 0.2,
            "segment_jsd_mean": 0.1,
        }
        s = structure_score(row)
        assert 0.0 <= s <= 1.0

    def test_structure_score_all_zeros_in_range(self):
        from signals import structure_score
        row = {k: 0.0 for k in [
            "entropy_norm", "autocorr_max", "spectral_concentration",
            "ar2_r2", "motif_top_share", "compression_ratio", "peak_freq_cv", "segment_jsd_mean"
        ]}
        assert 0.0 <= structure_score(row) <= 1.0

    def test_analyze_returns_dict(self):
        from signals import analyze
        result = analyze(self._sine_wave(), sr=256.0)
        assert isinstance(result, dict)
        for key in ["entropy_norm", "autocorr_max", "structure_score"]:
            assert key in result, f"missing key: {key}"

    def test_analyze_structure_score_in_range(self):
        from signals import analyze
        result = analyze(self._sine_wave(), sr=256.0)
        assert 0.0 <= result["structure_score"] <= 1.0

    def test_analyze_short_input(self):
        from signals import analyze
        result = analyze([0.1, 0.2, 0.3], sr=256.0)
        assert "structure_score" in result

    def test_bootstrap_ci(self):
        from signals import bootstrap_ci
        vals = [0.1 * i for i in range(20)]
        ci = bootstrap_ci(vals, seed=42, n_boot=100)
        assert "mean" in ci
        assert "ci_low" in ci
        assert "ci_high" in ci
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]

    def test_step_metrics(self):
        from signals import step_metrics
        t = [i * 0.01 for i in range(200)]
        y = [0.0 if i < 50 else 1.0 for i in range(200)]
        m = step_metrics(t, y)
        assert "rise_time" in m or "settling_time" in m

    # Additional coverage tests

    def test_compression_ratio_constant_signal(self):
        from signals import compression_ratio
        x = [0.5] * 512
        import numpy as np
        r = compression_ratio(np.array(x))
        assert 0.0 < r <= 1.0

    def test_spectral_analysis_sine(self):
        from signals import spectral_analysis
        import numpy as np
        x = np.array(self._sine_wave(n=512, freq=10.0, sr=256.0))
        pf, conc, ent = spectral_analysis(x, 256.0)
        assert pf >= 0.0
        assert 0.0 <= conc <= 1.0
        assert ent >= 0.0

    def test_motif_share_range(self):
        from signals import motif_share
        import numpy as np
        x = np.array(self._sine_wave())
        ms = motif_share(x)
        assert 0.0 <= ms <= 1.0

    def test_ar2_fit_returns_five_floats(self):
        from signals import ar2_fit
        import numpy as np
        x = np.array(self._sine_wave())
        a1, a2, r2, freq, rad = ar2_fit(x, 256.0)
        assert isinstance(a1, float)
        assert isinstance(a2, float)
        assert isinstance(r2, float)
        assert isinstance(freq, float)
        assert isinstance(rad, float)

    def test_generate_null_shuffle_same_length(self):
        from signals import generate_null
        import numpy as np
        x = np.array(self._sine_wave())
        null = generate_null(x, seed=0, kind="shuffle")
        assert len(null) == len(x)

    def test_generate_null_gaussian_same_length(self):
        from signals import generate_null
        import numpy as np
        x = np.array(self._sine_wave())
        null = generate_null(x, seed=0, kind="gaussian")
        assert len(null) == len(x)

    def test_generate_null_phase_scramble_same_length(self):
        from signals import generate_null
        import numpy as np
        x = np.array(self._sine_wave())
        null = generate_null(x, seed=0, kind="phase_scramble")
        assert len(null) == len(x)

    def test_generate_null_unknown_kind_raises(self):
        from signals import generate_null
        import numpy as np
        x = np.array(self._sine_wave())
        with pytest.raises(ValueError, match="Unknown null kind"):
            generate_null(x, seed=0, kind="unknown_kind")

    def test_bootstrap_ci_empty_values(self):
        from signals import bootstrap_ci
        ci = bootstrap_ci([], seed=0)
        assert math.isnan(ci["mean"])
        assert ci["n"] == 0

    def test_window_stats_short_signal(self):
        from signals import window_stats
        import numpy as np
        x = np.array([1.0, 2.0, 3.0])
        pcv, ecv, jsd = window_stats(x, sr=256.0, window=512)
        assert math.isnan(pcv)
        assert math.isnan(ecv)
        assert math.isnan(jsd)

    def test_analyze_all_keys_present(self):
        from signals import analyze
        result = analyze(self._sine_wave(n=512), sr=256.0)
        expected_keys = [
            "entropy_norm", "compression_ratio", "autocorr_lag1", "autocorr_max",
            "spectral_peak_hz", "spectral_concentration", "spectral_entropy",
            "motif_top_share", "peak_freq_cv", "entropy_cv", "segment_jsd_mean",
            "ar2_a1", "ar2_a2", "ar2_r2", "estimated_freq_hz", "root_radius",
            "structure_score"
        ]
        for k in expected_keys:
            assert k in result, f"missing key: {k}"
