"""MODULE_010_COHERENCE — runnable usage examples.

Demonstrates: shannon_entropy, mutual_information, global_coherence,
digitize, compute_metric_fields, build_coherence_passport,
seal_passport_hashes, and validate_coherence_passport.
"""
from __future__ import annotations

from gpts_core.coherence import (
    shannon_entropy,
    mutual_information,
    global_coherence,
    digitize,
    compute_metric_fields,
    build_coherence_passport,
    seal_passport_hashes,
    validate_coherence_passport,
)

# ---------------------------------------------------------------------------
# Example 1: Entropy and mutual information on discrete sequences
# ---------------------------------------------------------------------------

# Uniform binary: H = 1.0 bit
symbols_binary = [0, 1] * 50
h_binary = shannon_entropy(symbols_binary)
print(f"[Example 1] H(uniform binary) = {h_binary:.4f} bits  (expected ~1.0)")

# Certain: H = 0.0 bits
symbols_certain = [0] * 100
h_certain = shannon_entropy(symbols_certain)
print(f"[Example 1] H(certain) = {h_certain:.4f} bits  (expected 0.0)")

# Fully correlated sequences: high MI
x_corr = [0] * 50 + [1] * 50
y_corr = [0] * 50 + [1] * 50
mi_corr = mutual_information(x_corr, y_corr)
print(f"[Example 1] I(X;Y) fully correlated = {mi_corr:.4f} bits  (expected ~1.0)")

# Coherence ratio
gc = global_coherence(i_mutual=mi_corr, h_total=h_binary + h_binary)
print(f"[Example 1] global_coherence = {gc:.4f}  (range [0, 1])")

# ---------------------------------------------------------------------------
# Example 2: Digitize continuous values and compute metric fields
# ---------------------------------------------------------------------------

# Digitize a continuous ramp into 4 bins
ramp = [i * 0.1 for i in range(20)]
bins_out = digitize(ramp, bins=4)
print(f"\n[Example 2] digitize(ramp, bins=4): {bins_out}")

# Compute metric fields from two modules
observations = {
    "mod_A": [float(i) for i in range(20)],
    "mod_B": [float(i % 5) for i in range(20)],
}
fields = compute_metric_fields(observations, bins=8)
print(f"[Example 2] h_total = {fields['h_total']:.4f} bits")
print(f"[Example 2] i_mutual = {fields['i_mutual']:.4f} bits")
print(f"[Example 2] global_coherence = {fields['global_coherence']:.4f}")
print(f"[Example 2] inter_variance = {fields['inter_variance']:.4f}")

# ---------------------------------------------------------------------------
# Example 3: Build, seal, and validate a coherence passport
# ---------------------------------------------------------------------------

obs = {
    "sensor_A": [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1,
                 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.3],
    "sensor_B": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Build (hashes are None until sealed)
passport = build_coherence_passport(
    cycle_id="C99",
    module_observations=obs,
    system_version="LOCAL_CAPTURE",
    meta={"experiment": "example_3"},
)
print(f"\n[Example 3] passport verdict before sealing: {passport['verdict']}")
print(f"[Example 3] raw_payload_hash before sealing: {passport['raw_payload_hash']}")

# Seal to compute hashes
sealed = seal_passport_hashes(passport)
print(f"[Example 3] raw_payload_hash after sealing: {sealed['raw_payload_hash'][:40]}...")

# Validate the sealed passport
report = validate_coherence_passport(sealed)
print(f"[Example 3] validation verdict: {report['verdict']}")
print(f"[Example 3] replay_status: {report['replay_status']}")
print(f"[Example 3] formula_verified: {report['formula_verified']}")
print(f"[Example 3] errors: {report['errors']}")
