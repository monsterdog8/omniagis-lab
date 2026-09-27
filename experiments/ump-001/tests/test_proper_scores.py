import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_evaluator_adapter import score_trial
class TestScores(unittest.TestCase):
 def test_uniform_null_zero_skill(self):
  r=score_trial({"U0":1/3,"U1":1/3,"U2":1/3},"U0"); self.assertAlmostEqual(r["brier_skill"],0,places=15)
 def test_confident_wrong_penalized(self):
  r=score_trial({"U0":.98,"U1":.01,"U2":.01},"U2"); self.assertLess(r["brier_skill"],0); self.assertLess(r["log_skill"],0)
