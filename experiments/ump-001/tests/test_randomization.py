import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_core import assignment_from_digest_three_way,postcommit_assignment_digest
from ump_rc3_randomizer import derive_three_way
class TestRandomization(unittest.TestCase):
 def test_rejection_boundary(self): self.assertIsNone(assignment_from_digest_three_way(b"\xff"*32))
 def test_deterministic_binding(self):
  kw=dict(local_entropy=b"L"*32,future_external_value=b"future",trial_id="T",commit_hash_hex="00"*32)
  a=derive_three_way(**kw); b=derive_three_way(**kw); self.assertEqual(a,b)
 def test_commit_hash_changes_digest(self):
  common=dict(local_entropy=b"L"*32,future_external_value=b"future",trial_id="T")
  self.assertNotEqual(postcommit_assignment_digest(commit_hash_hex="00"*32,**common),postcommit_assignment_digest(commit_hash_hex="01"*32,**common))
