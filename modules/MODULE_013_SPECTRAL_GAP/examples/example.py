"""MODULE_013_SPECTRAL_GAP — usage examples.

Requires: numpy
Claim ceiling: LOCAL_ONLY__NO_EXTERNAL_PROOF
"""
import numpy as np
from gpts_core.spectral_gap import (
    pm_map,
    ulam_matrix,
    canonical_gap,
    run_pipeline,
    loglog_regression,
    BETA_PM,
    KAPPA_THEORY,
)

# ---------------------------------------------------------------------------
# Example 1: PM map evaluation and Ulam matrix for a single alpha
# ---------------------------------------------------------------------------

alpha = 0.10

# Evaluate the PM map on a sample of points
x = np.linspace(0.01, 0.99, 8)
y = pm_map(x, alpha=alpha)
print("Example 1: PM map evaluation")
print(f"  alpha = {alpha}, beta = {BETA_PM}")
for xi, yi in zip(x, y):
    print(f"  pm_map({xi:.2f}) = {yi:.6f}")

# Build the Ulam transition matrix
P, diag = ulam_matrix(alpha=alpha, n_bins=20, n_traj=5000, seed=42)
print(f"\n  Ulam matrix shape: {P.shape}")
print(f"  Row-stochastic error: {diag['row_stochastic_error']:.2e}")
print(f"  Zero rows: {diag['n_zero_rows']}")

# Compute the spectral gap
gap, aug_diag = canonical_gap(P, diag)
print(f"  Spectral gap: {gap}")
if aug_diag.get("FAIL"):
    print(f"  FAIL: {aug_diag['FAIL']}")
else:
    print(f"  lambda1_error: {aug_diag['lambda1_error']:.2e}")

# ---------------------------------------------------------------------------
# Example 2: Log-log regression on a synthetic gap sweep
# ---------------------------------------------------------------------------

print("\nExample 2: Log-log regression (synthetic data)")
alphas = [0.03, 0.05, 0.07, 0.10, 0.13, 0.17, 0.20]
gaps_synthetic = [0.02, 0.05, 0.09, 0.14, 0.21, 0.29, 0.38]

reg = loglog_regression(alphas, gaps_synthetic)
print(f"  kappa_theory = {KAPPA_THEORY:.4f}")
print(f"  kappa_fit    = {reg.get('kappa_fit')}")
print(f"  R2           = {reg.get('R2')}")
print(f"  qualificatif = {reg.get('qualificatif')}")
print(f"  FAIL         = {reg.get('FAIL')}")

# ---------------------------------------------------------------------------
# Example 3: Full pipeline over a small alpha grid
# ---------------------------------------------------------------------------

print("\nExample 3: Full pipeline (small grid for demonstration)")
alpha_grid = np.geomspace(0.03, 0.20, 8)
result = run_pipeline(
    alpha_grid,
    n_bins=40,
    n_traj=10_000,
    seed=42,
)
print(f"  alphas : {[round(a, 4) for a in result['alphas']]}")
print(f"  gaps   : {[round(g, 6) if g is not None else None for g in result['gaps']]}")
print(f"  kappa_fit    = {result['kappa_fit']}")
print(f"  R2           = {result['R2']}")
print(f"  D            = {result['D']}")
print(f"  qualificatif = {result['qualificatif']}")
print(f"  claim_ceiling: {result['claim_ceiling']}")
