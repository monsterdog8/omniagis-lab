# MODULE_013_SPECTRAL_GAP

**Source:** `gpts_core/spectral_gap.py`
**Version:** 1.0.0
**Coverage:** 81%
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF

## Description

Ulam spectral gap analysis for Pomeau-Manneville (PM) type-I intermittency maps.

Discretizes the PM map into a row-stochastic transition matrix (Ulam method), computes the canonical spectral gap `Delta = 1 - |lambda_2|`, and fits the log-log scaling law `log Delta = c + kappa * log alpha` to recover the universality exponent `kappa`.

## Function

- **Pomeau-Manneville map**: Vectorized type-I PM map with left branch `x + alpha * x^(1+beta)` and right branch `2x - 1`
- **Ulam transition matrix**: Row-stochastic discretization of the PM map dynamics
- **Spectral gap computation**: Eigenvalue analysis with fail-closed guards
- **Log-log regression**: OLS regression for the kappa scaling exponent with qualification

## Constants

| Name | Value | Description |
|------|-------|-------------|
| `BETA_PM` | 0.75 | Default PM map exponent |
| `KAPPA_THEORY` | ~1.667 | Theoretical kappa = 2/(2-beta) |
| `MIN_GAP` | 5e-5 | Minimum gap for regression |
| `N_VALID_MIN` | 6 | Minimum valid points required |
| `R2_GOOD` | 0.97 | Good scaling R2 threshold |
| `R2_PUB` | 0.985 | Publication-grade R2 threshold |

## Public API

```python
pm_map(x: np.ndarray, alpha: float, beta: float = BETA_PM) -> np.ndarray
ulam_matrix(alpha: float, n_bins=80, n_traj=80_000, seed=42, beta=BETA_PM) -> Tuple[np.ndarray, Dict]
canonical_gap(P: np.ndarray, diag: Dict) -> Tuple[Optional[float], Dict]
compute_gap_point(alpha: float, n_bins=80, n_traj=80_000, seed=42, beta=BETA_PM) -> Tuple[Optional[float], Dict]
loglog_regression(alphas, gaps, min_gap=MIN_GAP, n_valid_min=N_VALID_MIN) -> Dict
run_pipeline(alpha_grid, n_bins=80, n_traj=80_000, seed=42, beta=BETA_PM, min_gap=MIN_GAP) -> Dict
```

## Dependencies

- **stdlib:** `math`, `typing`
- **third_party:** `numpy`

## Files

```
MODULE_013_SPECTRAL_GAP/
  manifest.json
  README.md
  schema.json
  tests/test_MODULE_013.py
  examples/example.py
  exports/spectral_gap.py
```

## Status

- **Status:** ACTIVE
- **Tests:** `tests/test_MODULE_013.py`
- **Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF
