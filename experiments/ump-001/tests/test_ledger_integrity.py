import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ump_rc3_commit import chain_entry
class TestLedger(unittest.TestCase):
 def test_chain_changes_with_prev(self):
  e={"schema":"x","seq":1,"trial_id":"T","event":"X","wall_time_utc":"2026-01-01T00:00:00Z","monotonic_ns":1,"actor":"TEST","payload_sha256":"00"*32}
  a=chain_entry(e,"00"*32); b=chain_entry(e,"01"*32)
  self.assertNotEqual(a["entry_sha256"],b["entry_sha256"])
