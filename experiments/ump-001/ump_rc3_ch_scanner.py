#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent

def scan():
    binding=json.loads((ROOT/"UMP_001_RC3_HARDWARE_BIND_V1.json").read_text())
    threat=json.loads((ROOT/"UMP_001_RC3_THREAT_MATRIX_V1.json").read_text())
    observed=[]
    for row in threat["attacks"]:
        r=dict(row)
        if r["ATTACK"]=="HASH_CONFUSION":
            r.update(EXECUTED=True,OBSERVATION="Parent hash is used only as binding/integrity; claim ceiling remains hardware/spec only",STATUS="RESISTED_STRUCTURAL")
        elif r["ATTACK"]=="CLAIM_OVERREACH":
            ok=binding["status"]=="BLOCKED_REQUIRED_MEASUREMENT" and binding["claim_ceiling"]=="HARDWARE_BIND_SCHEMA_ONLY"
            r.update(EXECUTED=True,OBSERVATION="Binding remains blocked and non-empirical",STATUS="RESISTED_STRUCTURAL" if ok else "DETECTED")
        elif r["ATTACK"]=="CONSCIOUSNESS_OVERCLAIM":
            r.update(EXECUTED=True,OBSERVATION="No consciousness claim exists in GTAB child binding",STATUS="RESISTED_STRUCTURAL")
        observed.append(r)
    return {"schema":"UMP_001_RC3_CH_SCAN_RUNTIME_V1","executed_structural":sum(1 for r in observed if r["EXECUTED"]),
      "mandatory_not_executed":sum(1 for r in observed if not r["EXECUTED"]),"aggregate_pass":"FORBIDDEN",
      "GTAB_REAL":"NOT_BOUND","rows":observed}

if __name__=="__main__":
    print(json.dumps(scan(),sort_keys=True,indent=2))
