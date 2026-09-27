import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_core import commit_prediction,verify_commitment
class TestCommitReveal(unittest.TestCase):
 def test_round_trip_and_mutation(self):
  p={"trial_id":"T1","prediction_distribution":{"U0":.2,"U1":.3,"U2":.5}}; n=bytes(range(32))
  h=commit_prediction(p,n); self.assertTrue(verify_commitment(p,n,h))
  q=dict(p); q["trial_id"]="T2"; self.assertFalse(verify_commitment(q,n,h))
