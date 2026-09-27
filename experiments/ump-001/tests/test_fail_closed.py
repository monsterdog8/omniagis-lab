import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_acquisition import preflight
from ump_rc3_runner import run_real_trial
class TestFailClosed(unittest.TestCase):
 def test_preflight_blocked(self): self.assertEqual(preflight()["status"],"BLOCKED_REQUIRED_MEASUREMENT")
 def test_real_trial_refuses(self):
  with self.assertRaisesRegex(RuntimeError,"MISSING_PHYSICAL_BINDINGS"): run_real_trial()
