# MODULE_004_GATE

Claim classification and evidence scoring for the OmniAGIS audit pipeline.

## Function

Implements a fail-closed evidence gating system with two main capabilities:

1. **Claim classifier**: Regex-based detection of forbidden strong claims (BLOCKED) vs.
   prudent bounded hypotheses (ALLOWED_BOUNDED) vs. unreviewed scope (UNKNOWN_REQUIRES_REVIEW).

2. **Evidence scorer**: Weighted score across six dimensions (DATA, RAW, SCORING, REPLAY,
   INDEPENDENCE, SAFETY). Score is zero unless RAW outputs are present and valid.
   Public claims require `final_score >= 0.85` AND all dimensions non-zero AND
   `SAFETY >= 0.75` AND `independent_review=True`.

Additional utilities:
- **`maturity_map`**: Tracks invention maturity through 8 stages (IDEA → DEPLOY).
  Public claim right is always BLOCKED regardless of maturity.
- **`proof_firewall`**: Final gate returning BLOCKED unless all proof dimensions are
  non-zero and SAFETY >= 0.75.

## Source

`gpts_core/gate.py` — copied verbatim to `exports/gate.py`

## Public API

| Symbol | Type | Description |
|---|---|---|
| `ClaimStatus` | `Enum` | `BLOCKED` / `ALLOWED_BOUNDED` / `UNKNOWN_REQUIRES_REVIEW` |
| `ClaimResult` | dataclass | Claim classification result |
| `classify_claim(text)` | function | Classify a text claim |
| `EVIDENCE_WEIGHTS` | `Dict[str, float]` | Per-dimension weights summing to 1.0 |
| `MATURITY_STAGES` | `List[str]` | 8 maturity stage names |
| `EvidenceInput` | dataclass | Input to `compute_evidence_score` |
| `EvidenceScore` | dataclass | Weighted evidence score result |
| `compute_evidence_score(inp)` | function | Compute evidence score with penalties |
| `from_gate_counts(...)` | function | Construct `EvidenceInput` from raw counts |
| `maturity_map(**flags)` | function | Track maturity stage |
| `proof_firewall(dims)` | function | Final public-claim gate |

### Evidence dimension weights

```
DATA         0.20
RAW          0.20
SCORING      0.20
REPLAY       0.15
INDEPENDENCE 0.15
SAFETY       0.10
```

### `ClaimResult` fields

```
text: str
status: str          # ClaimStatus value
reason: str
hits: List[str]      # matched pattern names
required_next_gate: str
to_dict() -> Dict
```

### `EvidenceScore` fields

```
dimensions: Dict[str, float]
weighted_raw: float
penalties: List[str]
final_score: float
scoring_status: str    # READY_FOR_SCORING | SCORING_DONE_LOCAL | BLOCKED
public_claim_allowed: bool
verdict: str
notes: List[str]
to_dict() -> Dict
```

## Dependencies

- `dataclasses` (stdlib)
- `enum` (stdlib)
- `re` (stdlib)

## Coverage

91% (as measured against `tests/test_gpts_core.py::TestGate`)

## Claim Ceiling

`LOCAL_ONLY__NO_EXTERNAL_PROOF` — evidence scoring is local; public claims require external independent review not provided by this module.

## Usage

```python
from modules.MODULE_004_GATE.exports.gate import classify_claim, from_gate_counts, compute_evidence_score

# Classify a text claim
result = classify_claim("This is a hypothesis, might be testable as LAB_ONLY prototype")
print(result.status)  # ALLOWED_BOUNDED

# Score evidence
inp = from_gate_counts(expected=10, present=10, valid=10, safety=0.9,
                       scoring=True, replay=True, independence=True)
score = compute_evidence_score(inp)
print(score.final_score, score.public_claim_allowed)
```

See `examples/example.py` for runnable examples.
