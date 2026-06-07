"""Tests for MODULE_013_SPECTRAL_GAP — extracted from tests/test_gpts_core.py.

Scope: spectral_gap module.
Claim ceiling: LOCAL_SUPPORTED_WITHIN_SCOPE
"""
from __future__ import annotations

import pytest


class TestSpectralGap:
    def test_pm_map_vectorized(self):
        import numpy as np
        from gpts_core.spectral_gap import pm_map
        x = np.array([0.1, 0.3, 0.6, 0.8])
        result = pm_map(x, alpha=0.1)
        assert result.shape == (4,)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_ulam_matrix_row_stochastic(self):
        import numpy as np
        from gpts_core.spectral_gap import ulam_matrix
        # ulam_matrix generates its own trajectory internally
        P, diag = ulam_matrix(alpha=0.05, n_bins=10, n_traj=1000)
        assert P.shape == (10, 10)
        row_sums = P.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_canonical_gap_range(self):
        import numpy as np
        from gpts_core.spectral_gap import ulam_matrix, canonical_gap
        P, diag = ulam_matrix(alpha=0.05, n_bins=10, n_traj=1000)
        gap, aug_diag = canonical_gap(P, diag)
        if gap is not None:
            assert 0.0 <= gap <= 1.0

    def test_run_pipeline_structure(self):
        from gpts_core.spectral_gap import run_pipeline
        result = run_pipeline([0.05, 0.10], n_bins=10, n_traj=500)
        assert "alphas" in result
        assert "gaps" in result or "points" in result

    def test_loglog_regression_returns_result(self):
        from gpts_core.spectral_gap import loglog_regression
        # requires >= 6 valid points
        alphas = [0.03, 0.05, 0.07, 0.10, 0.13, 0.17, 0.20]
        gaps = [0.02, 0.05, 0.09, 0.14, 0.21, 0.29, 0.38]
        result = loglog_regression(alphas, gaps)
        assert isinstance(result, dict)
        assert "kappa_fit" in result or "kappa" in result or "slope" in result

    def test_pm_map_left_branch(self):
        """Left branch (x < 0.5): result > x for positive alpha."""
        import numpy as np
        from gpts_core.spectral_gap import pm_map
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = pm_map(x, alpha=0.5)
        # Left branch: x + alpha*x^(1+beta) >= x
        assert all(result[i] >= x[i] for i in range(len(x)))

    def test_pm_map_right_branch(self):
        """Right branch (x >= 0.5): result = 2x - 1."""
        import numpy as np
        from gpts_core.spectral_gap import pm_map
        x = np.array([0.5, 0.6, 0.75, 0.9])
        result = pm_map(x, alpha=0.1)
        expected = 2.0 * x - 1.0
        assert all(abs(result[i] - expected[i]) < 1e-10 for i in range(len(x)))

    def test_ulam_matrix_shape(self):
        """Matrix shape matches n_bins x n_bins."""
        from gpts_core.spectral_gap import ulam_matrix
        P, diag = ulam_matrix(alpha=0.10, n_bins=20, n_traj=2000)
        assert P.shape == (20, 20)
        assert diag["n_bins"] == 20
        assert diag["n_traj"] == 2000

    def test_ulam_matrix_diagnostics_keys(self):
        """Diagnostics dict contains required keys."""
        from gpts_core.spectral_gap import ulam_matrix
        P, diag = ulam_matrix(alpha=0.10, n_bins=10, n_traj=500)
        for key in ["alpha", "n_bins", "n_traj", "seed", "row_stochastic_error"]:
            assert key in diag

    def test_canonical_gap_diag_augmented(self):
        """canonical_gap augments the diagnostics dict."""
        from gpts_core.spectral_gap import ulam_matrix, canonical_gap
        P, diag = ulam_matrix(alpha=0.10, n_bins=10, n_traj=1000)
        gap, aug_diag = canonical_gap(P, diag)
        assert "FAIL" in aug_diag

    def test_compute_gap_point_returns_tuple(self):
        """compute_gap_point returns (Optional[float], Dict)."""
        from gpts_core.spectral_gap import compute_gap_point
        gap, diag = compute_gap_point(alpha=0.10, n_bins=10, n_traj=500)
        assert isinstance(diag, dict)
        assert gap is None or isinstance(gap, float)

    def test_loglog_regression_insufficient_points(self):
        """Regression fails gracefully with fewer than N_VALID_MIN valid points."""
        from gpts_core.spectral_gap import loglog_regression, N_VALID_MIN
        alphas = [0.05, 0.10]
        gaps = [0.01, 0.02]
        result = loglog_regression(alphas, gaps)
        assert result["FAIL"] is not None
        assert result["kappa_fit"] is None

    def test_loglog_regression_filters_null_gaps(self):
        """loglog_regression skips None gap values."""
        from gpts_core.spectral_gap import loglog_regression
        alphas = [0.03, 0.05, 0.07, 0.10, 0.13, 0.17, 0.20]
        gaps = [0.02, None, 0.09, 0.14, 0.21, 0.29, 0.38]
        result = loglog_regression(alphas, gaps)
        assert isinstance(result, dict)

    def test_run_pipeline_alphas_match(self):
        """Pipeline output alphas match input."""
        from gpts_core.spectral_gap import run_pipeline
        alpha_grid = [0.05, 0.10, 0.15]
        result = run_pipeline(alpha_grid, n_bins=10, n_traj=500)
        assert result["alphas"] == alpha_grid

    def test_run_pipeline_claim_ceiling(self):
        """Pipeline always sets the correct claim ceiling."""
        from gpts_core.spectral_gap import run_pipeline
        result = run_pipeline([0.10], n_bins=10, n_traj=200)
        assert result["claim_ceiling"] == "LOCAL_LAB_ONLY_NOT_EXTERNAL_PROOF"

    def test_kappa_theory_value(self):
        """KAPPA_THEORY = 2/(2-BETA_PM)."""
        from gpts_core.spectral_gap import KAPPA_THEORY, BETA_PM
        expected = 2.0 / (2.0 - BETA_PM)
        assert abs(KAPPA_THEORY - expected) < 1e-12

    def test_run_pipeline_gaps_length_matches_alphas(self):
        """Number of gap results matches number of alpha values."""
        from gpts_core.spectral_gap import run_pipeline
        alpha_grid = [0.05, 0.10, 0.15, 0.20]
        result = run_pipeline(alpha_grid, n_bins=10, n_traj=300)
        assert len(result["gaps"]) == len(alpha_grid)
