#!/usr/bin/env python3
from __future__ import annotations
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
EXPECTED_PARENT="3d000c5ec1ec27d1fde68fa528ac25e9d5c34b698e3cc7f684682cf180f3057c"
REQUIRED_FILES=[
"GTAB_BIND_001.json","UMP_001_RC3_HARDWARE_BIND_V1.json","UMP_001_RC3_CALIBRATION_RECEIPT_V1.json",
"UMP_001_RC3_DEPTH_HORIZON_FREEZE_V1.json","UMP_001_RC3_PILOT_PREREGISTRATION_V1.json",
"UMP_001_RC3_ENGINEERING_PILOT_CONFIG_V1.json","UMP_001_RC3_CHANNEL_AUDIT_V1.json",
"UMP_001_RC3_RANDOMIZATION_RECEIPT_V1.json","UMP_001_RC3_LEDGER_SCHEMA_V1.json",
"UMP_001_RC3_THREAT_MATRIX_V1.json","UMP_001_RC3_POWER_PLAN_V1.json",
"ump_rc3_core.py","ump_rc3_acquisition.py","ump_rc3_commit.py","ump_rc3_randomizer.py",
"ump_rc3_evaluator_adapter.py","ump_rc3_power_plan.py","ump_rc3_runner.py","ump_rc3_ch_scanner.py"
]
def load(n): return json.loads((ROOT/n).read_text(encoding="utf-8"))
def require(c,m):
 if not c: raise AssertionError(m)
def main():
 missing=[f for f in REQUIRED_FILES if not (ROOT/f).exists()]
 require(not missing,f"missing child artifacts: {missing}")
 binding=load("GTAB_BIND_001.json"); hw=load("UMP_001_RC3_HARDWARE_BIND_V1.json")
 cal=load("UMP_001_RC3_CALIBRATION_RECEIPT_V1.json"); dh=load("UMP_001_RC3_DEPTH_HORIZON_FREEZE_V1.json")
 pilot=load("UMP_001_RC3_ENGINEERING_PILOT_CONFIG_V1.json"); prereg=load("UMP_001_RC3_PILOT_PREREGISTRATION_V1.json")
 channel=load("UMP_001_RC3_CHANNEL_AUDIT_V1.json"); rand=load("UMP_001_RC3_RANDOMIZATION_RECEIPT_V1.json")
 power=load("UMP_001_RC3_POWER_PLAN_V1.json"); sync=load("UMP_001_HANGAR_SYNC_001.json")
 require(binding["parent"]["stage0_capsule_sha256"]==EXPECTED_PARENT,"binding parent drift")
 require(binding["parent"]["immutable"] is True,"Stage0 parent must remain immutable")
 require(binding["status"]=="PROPOSED_NOT_EXECUTED","GTAB_BIND overpromoted")
 require(binding["gates"]["EMPIRICAL_C2_EVIDENCE"]==0 and binding["gates"]["EMPIRICAL_C4_EVIDENCE"]==0,"empirical evidence must remain zero")
 require(binding["gates"]["OMEGA_U"]=="NOT_MEASURED","Omega_U overpromoted")
 require(hw["parent_stage0_capsule_sha256"]==EXPECTED_PARENT and hw["status"]=="BLOCKED_REQUIRED_MEASUREMENT","hardware bind must remain blocked")
 require(cal["status"]=="NOT_EXECUTED_BLOCKED_HARDWARE","calibration must remain not executed")
 require(dh["C4"]["D_PRIMARY"]=="TBD" and dh["C4"]["DELTA_PRIMARY"]=="TBD","D*/Delta* must remain TBD")
 require(prereg["pilot_authorized"] is False,"pilot prereg may not authorize execution")
 require(pilot["readiness_rule"]["pilot_authorized"] is False and pilot["pilot_design"]["confirmatory_reuse_forbidden"] is True,"pilot config drift")
 require(all(s["status"]=="NOT_EXECUTED" for s in channel["stages"]),"channel closure fabricated")
 require(rand["postcommit_physical_independence"]=="NOT_PROVEN","physical RNG independence overpromoted")
 require(power["canonical_confirmatory_N"]=="NOT_COMPUTED","confirmatory N fabricated")
 require(sync["methodological_sync"]["universal_connection"]=="NOT_SUPPORTED","universal connection overpromoted")
 print("UMP_001_CHILD_STRUCTURAL_VALIDATION_PASS")
 return 0
if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception as exc:
  print(f"UMP_001_CHILD_STRUCTURAL_VALIDATION_FAIL: {exc}",file=sys.stderr); raise
