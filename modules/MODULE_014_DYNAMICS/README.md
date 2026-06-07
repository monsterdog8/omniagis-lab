# MODULE_014_DYNAMICS

**Source:** `gpts_core/dynamics.py`
**Version:** 1.0.0
**Coverage:** 82%
**Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF

## Description

Coupled state-variable dynamics and fractal coherence tracking. Contains two independent models:

1. **ContinuumState / step_continuum**: 6-variable discrete-time nonlinear dynamics system modelling coherence, fusion, entropy, chaos, resonance, and energy.
2. **FractalEngine / FractalState**: Cycle-based coherence tracking with deterministic SHA-256 drift and time-based sine modulation. Pure stdlib — no external dependencies.

## Function

- **Continuum dynamics**: 6-variable coupled system advancing via `step_continuum()` with Gaussian noise. Variables psi, F, S, C are clamped to [0,1]. R and Q are derived.
- **Fractal coherence engine**: Per-cycle coherence computation using SHA-256 hash of cycle_id for deterministic drift. Tracks history, computes mean/std/min/max, and evaluates stability.

## ContinuumState Fields

| Field | Default | Description |
|-------|---------|-------------|
| `psi` | 0.999 | Coherence [0,1] |
| `F` | 0.90 | Fusion [0,1] |
| `S` | 0.0001 | Entropy [0,1] |
| `C` | 0.15 | Chaos [0,1] |
| `R` | 55.435 | Resonance Hz |
| `Q` | 0.0 | Energy (derived) |

## FractalState Fields

| Field | Default | Description |
|-------|---------|-------------|
| `coherence` | 1.0 | Signal coherence psi (target 1.0) |
| `entropy` | 0.0 | 1 - coherence |
| `resonance_hz` | 11.987 | Current resonance frequency (Hz) |
| `drift` | 0.0 | Deviation range from coherence history |

## Public API

```python
# Continuum dynamics
ContinuumState(psi=0.999, F=0.90, S=0.0001, C=0.15, R=55.435, Q=0.0)
energy(state: ContinuumState) -> float
step_continuum(state, *, noise_std=0.002, rng=None) -> ContinuumState
run_continuum(n_steps, initial=None, *, noise_std=0.002, seed=None) -> List[Dict]

# Fractal coherence engine
FractalState(coherence=1.0, entropy=0.0, resonance_hz=11.987, drift=0.0)
FractalEngine(history_size=100)
FractalEngine.compute_metrics(cycle_id=None) -> FractalState
FractalEngine.get_state() -> FractalState
FractalEngine.get_history() -> List[float]
FractalEngine.get_statistics() -> Dict[str, float]
FractalEngine.is_stable(drift_threshold=0.0001, coherence_threshold=0.999) -> bool
FractalEngine.reset() -> None
```

## Step Equations (dt=1)

```
psi' = psi + a1*(F - S) - b1*C + noise
F'   = F   + a2*psi*(1-F) - b2*S*F + g2*C*(1-F) + noise
S'   = S   + a3*C*(1-S) - b3*psi*S - k3*F*S + noise
C'   = C   + a4*(1-C)*max(0, 1-psi) - b4*C*F + noise
R    = R_base + d*(F - S)
Q    = (F * psi) / (1 + S) * R
```

## Dependencies

- **stdlib:** `math`, `hashlib`, `random`, `time`, `collections`, `dataclasses`, `typing`
- **third_party:** none

## Files

```
MODULE_014_DYNAMICS/
  manifest.json
  README.md
  schema.json
  tests/test_MODULE_014.py
  examples/example.py
  exports/dynamics.py
```

## Status

- **Status:** ACTIVE
- **Tests:** `tests/test_MODULE_014.py`
- **Claim ceiling:** LOCAL_ONLY__NO_EXTERNAL_PROOF
