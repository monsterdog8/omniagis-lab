# MODULE_001_SIGNALS

Time-series structure analysis for the OmniAGIS audit pipeline.

## Function

Computes a comprehensive set of structure metrics for 1-D time-series signals:
- **Entropy**: Normalized Shannon entropy of the amplitude histogram
- **Compression**: zlib compression ratio as a structure proxy (lower = more compressible = more structure)
- **Autocorrelation**: Lag-1 and max-absolute autocorrelation over a configurable lag window
- **Spectral**: Peak frequency (Hz), spectral concentration, spectral entropy via Hanning-windowed FFT
- **Motif**: Fraction of the most common ordinal length-4 motif
- **Window statistics**: Coefficient-of-variation for peak frequency and entropy across non-overlapping windows, plus mean Jensen-Shannon divergence between adjacent windows
- **AR(2) fit**: Autoregressive order-2 model coefficients, R², estimated frequency, root radius
- **Structure score**: Weighted composite score in [0, 1]
- **Step-response**: Initial slope, rise time, overshoot, settling time, steady-state value
- **Null models**: shuffle / gaussian / same_marginal / phase_scramble surrogates
- **Bootstrap CI**: 95% confidence interval for any list of finite values

## Source

`gpts_core/signals.py` — copied verbatim to `exports/signals.py`

## Public API

| Function | Signature | Returns |
|---|---|---|
| `entropy_norm` | `(x, bins=32)` | `float` |
| `compression_ratio` | `(x)` | `float` |
| `autocorr` | `(x, maxlag=64)` | `Tuple[float, float]` |
| `spectral_analysis` | `(x, sr)` | `Tuple[float, float, float]` |
| `motif_share` | `(x, alphabet=8, mlen=4)` | `float` |
| `window_stats` | `(x, sr, window=512)` | `Tuple[float, float, float]` |
| `ar2_fit` | `(x, sr)` | `Tuple[float, float, float, float, float]` |
| `structure_score` | `(row)` | `float` |
| `analyze` | `(x, sr=256.0)` | `Dict[str, float]` |
| `step_metrics` | `(t, y, eps=0.02)` | `Dict[str, float]` |
| `generate_null` | `(x, seed, kind)` | `np.ndarray` |
| `bootstrap_ci` | `(values, seed, n_boot=200)` | `Dict[str, float]` |

## Dependencies

- `numpy >= 1.21`
- `scipy >= 1.7`

## Coverage

69% (as measured against `tests/test_gpts_core.py::TestSignals`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — all metrics are locally computed; no external validation or publication has been performed.

## Usage

```python
from modules.MODULE_001_SIGNALS.exports.signals import analyze

signal = [0.1 * i for i in range(512)]
metrics = analyze(signal, sr=256.0)
print(metrics["structure_score"])
```

See `examples/example.py` for runnable examples.
