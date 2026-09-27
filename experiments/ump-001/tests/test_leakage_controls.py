import json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestLeakageControls(unittest.TestCase):
 def test_staircase_is_not_falsely_closed(self):
  c=json.loads((ROOT/"UMP_001_RC3_CHANNEL_AUDIT_V1.json").read_text())
  self.assertEqual(len(c["stages"]),9); self.assertTrue(c["leakage_canary_required"])
  self.assertTrue(all(s["status"]=="NOT_EXECUTED" and not s["verified"] for s in c["stages"]))
