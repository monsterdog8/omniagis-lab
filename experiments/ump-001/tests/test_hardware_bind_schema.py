import json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestHardwareBind(unittest.TestCase):
 def test_required_fields_present_and_uninvented(self):
  b=json.loads((ROOT/"UMP_001_RC3_HARDWARE_BIND_V1.json").read_text()); p=b["physical_bind"]
  req=["OBSERVATION_DEVICE_MANUFACTURER","OBSERVATION_DEVICE_MODEL","OBSERVATION_PRECISION","SAMPLING_RATE_HZ","MEASURED_R_OHM","MEASURED_C_FARAD","VOLTAGE_U0","VOLTAGE_U1","VOLTAGE_U2","SAFE_VOLTAGE_RANGE","ACQUISITION_PATH","INTERVENTION_LATENCY"]
  self.assertTrue(all(k in p for k in req)); self.assertEqual(b["status"],"BLOCKED_REQUIRED_MEASUREMENT")
  self.assertTrue(any(p[k] in ("TBD","NOT_MEASURED") for k in req))
