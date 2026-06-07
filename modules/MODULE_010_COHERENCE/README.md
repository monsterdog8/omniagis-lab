# MODULE_010_COHERENCE

## Purpose

Coherence metrics: Shannon entropy, mutual information, and replayable coherence passports.

Computes module-level entropy, joint entropy, and mutual information from discretized numeric observations. Builds and validates coherence passports with SHA-256 hash chains. All entropy computations use base-2 bits. All claims are LOCAL_LAB_ONLY — not external proof.

## Source

`gpts_core/coherence.py`

## Public API

| Function | Signature | Returns | Description |
|---|---|---|---|
| `shannon_entropy` | `(discrete: Sequence[int]) -> float` | `float` | H(X) in bits |
| `joint_entropy` | `(x: Sequence[int], y: Sequence[int]) -> float` | `float` | H(X,Y) in bits |
| `mutual_information` | `(x: Sequence[int], y: Sequence[int]) -> float` | `float` | I(X;Y) clipped to [0, inf) |
| `global_coherence` | `(i_mutual: float, h_total: float) -> float` | `float` | C(S) = I/H, 0 if H=0 |
| `digitize` | `(values: Sequence[float], bins: int = 8) -> List[int]` | `List[int]` | Continuous to discrete bins |
| `compute_metric_fields` | `(observations: Dict[str, List[float]], bins: int = 8) -> Dict[str, Any]` | `Dict[str, Any]` | Full metric field computation |
| `normalize_observations` | `(module_observations: Any) -> Tuple[Dict, List[str], List[str]]` | `Tuple` | Validate and normalize observations |
| `build_coherence_passport` | `(cycle_id, module_observations, ...) -> Dict` | `Dict` | Build unsealed coherence passport |
| `seal_passport_hashes` | `(passport: Dict) -> Dict` | `Dict` | Fill raw_payload_hash and entry_hash |
| `validate_coherence_passport` | `(passport: Dict, *, tolerance: float = 1e-9) -> Dict` | `Dict` | Validate sealed passport |

## Formula

```
C(S) = I_mutual(S) / H_total(S)
```

Where:
- `I_mutual(S)` = sum of upper-triangle pairwise mutual information
- `H_total(S)` = sum of per-module Shannon entropy (base-2 bits)
- Observations are discretized into `bins` equal-width bins

## Dependencies

All stdlib: `math`, `hashlib`, `json`, `copy`, `datetime`

## Coverage

78%

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF`

All coherence values are local heuristic computations on discretized samples. No calibration or external benchmark is claimed.

## Usage

See `examples/example.py` for runnable usage examples.

## Tests

`tests/test_MODULE_010.py` — extracted from `tests/test_gpts_core.py::TestCoherence`
