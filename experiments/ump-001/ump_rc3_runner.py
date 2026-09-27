#!/usr/bin/env python3
from __future__ import annotations
import json
from ump_rc3_acquisition import preflight

def canonical_preflight():
    p=preflight()
    p.update({
      "program":"UMP-001","gtab_id":"GTAB_RC3_001",
      "EMPIRICAL_C2_EVIDENCE":0,"EMPIRICAL_C4_EVIDENCE":0,
      "OMEGA_U":"NOT_MEASURED","POSTCOMMIT_PHYSICAL_INDEPENDENCE":"NOT_PROVEN",
      "UNIVERSAL_CONNECTION":"NOT_SUPPORTED"
    })
    return p

def run_real_trial():
    p=canonical_preflight()
    if p["status"]!="READY_FOR_ENGINEERING_CALIBRATION":
        raise RuntimeError("REAL_TRIAL_BLOCKED_MISSING_PHYSICAL_BINDINGS")
    raise RuntimeError("REAL_TRIAL_NOT_IMPLEMENTED_UNTIL_CALIBRATION_AND_FREEZE")

if __name__=="__main__":
    print(json.dumps(canonical_preflight(),sort_keys=True,indent=2))
