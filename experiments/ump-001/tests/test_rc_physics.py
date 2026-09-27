import pathlib,sys,unittest,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_core import rc_voltage
class TestRCPhysics(unittest.TestCase):
 def test_t0(self): self.assertAlmostEqual(rc_voltage(r_ohm=1000,c_farad=.001,v0=1.2,vin=3.3,delta_s=0),1.2,places=12)
 def test_tau(self):
  v=rc_voltage(r_ohm=1000,c_farad=.001,v0=0,vin=1,delta_s=1)
  self.assertAlmostEqual(v,1-math.exp(-1),places=12)
 def test_domain(self):
  with self.assertRaises(ValueError): rc_voltage(r_ohm=0,c_farad=.001,v0=0,vin=1,delta_s=1)
