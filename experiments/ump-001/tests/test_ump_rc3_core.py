#!/usr/bin/env python3
import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ump_rc3_core import (
    GateState,
    assignment_from_digest_three_way,
    commit_prediction,
    postcommit_assignment_digest,
    proper_score_skill,
    rc_voltage,
    uniform_null,
    verify_commitment,
)

class UMP001RC3Tests(unittest.TestCase):
    def test_rc_boundary(self):
        self.assertAlmostEqual(rc_voltage(r_ohm=1000, c_farad=0.001, v0=1.2, vin=3.3, delta_s=0), 1.2, places=12)

    def test_rc_asymptote(self):
        v = rc_voltage(r_ohm=1000, c_farad=0.001, v0=0.0, vin=3.3, delta_s=20.0)
        self.assertGreater(v, 3.299999)

    def test_uniform_three_state_brier_null(self):
        null = uniform_null(["A","B","C"])
        s = proper_score_skill(null, null, "A")
        self.assertAlmostEqual(s["brier_skill"], 0.0, places=15)
        self.assertAlmostEqual(s["negative_brier_null"], -(2/3), places=15)

    def test_confident_wrong_is_penalized(self):
        null = uniform_null(["A","B","C"])
        model = {"A":0.98,"B":0.01,"C":0.01}
        s = proper_score_skill(model, null, "C")
        self.assertLess(s["brier_skill"], 0.0)
        self.assertLess(s["log_skill"], 0.0)

    def test_commit_reveal_round_trip(self):
        payload = {
            "protocol_version":"UMP-001-child-v1",
            "trial_id":"trial_000001",
            "probabilities":{"A":0.2,"B":0.3,"C":0.5},
        }
        nonce = bytes(range(32))
        h = commit_prediction(payload, nonce)
        self.assertTrue(verify_commitment(payload, nonce, h))
        modified = dict(payload)
        modified["trial_id"] = "trial_000002"
        self.assertFalse(verify_commitment(modified, nonce, h))

    def test_three_way_rejection_removes_modulo_bias_boundary(self):
        self.assertIsNone(assignment_from_digest_three_way(b"\xff"*32))
        self.assertIn(assignment_from_digest_three_way(b"\x00"*32), (0,1,2))

    def test_postcommit_digest_binds_commit_hash(self):
        common = dict(
            local_entropy=b"L"*32,
            future_external_value=b"future-pulse",
            trial_id="trial_42",
        )
        d1 = postcommit_assignment_digest(commit_hash_hex="00"*32, **common)
        d2 = postcommit_assignment_digest(commit_hash_hex="01"*32, **common)
        self.assertNotEqual(d1,d2)

    def test_gate_state_fail_closed(self):
        g = GateState()
        self.assertFalse(g.engineering_pilot_ready)
        self.assertFalse(g.c4_confirmatory_ready)
        ready = GateState(
            gtab_real_bound=True,
            observation_device_bound=True,
            precision_bound=True,
            sampling_bound=True,
            d_primary_bound=True,
            delta_primary_bound=True,
            physical_rng_independence_proven=False,
        )
        self.assertTrue(ready.engineering_pilot_ready)
        self.assertFalse(ready.c4_confirmatory_ready)

if __name__ == "__main__":
    unittest.main(verbosity=2)
