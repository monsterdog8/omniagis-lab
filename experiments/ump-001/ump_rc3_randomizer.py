#!/usr/bin/env python3
from __future__ import annotations
import secrets
from ump_rc3_core import postcommit_assignment_digest, assignment_from_digest_three_way

def derive_three_way(*, local_entropy, future_external_value, trial_id, commit_hash_hex):
    digest=postcommit_assignment_digest(local_entropy=local_entropy,future_external_value=future_external_value,
        trial_id=trial_id,commit_hash_hex=commit_hash_hex)
    assignment=assignment_from_digest_three_way(digest)
    if assignment is None:
        return {"status":"REJECT_AND_REDRAW","digest_hex":digest.hex(),"assignment":None}
    return {"status":"ASSIGNED","digest_hex":digest.hex(),"assignment":assignment}

def local_selftest_assignment(*, trial_id, commit_hash_hex):
    # Same-process CSPRNG is a software selftest only; it does NOT prove separate-host or physical independence.
    entropy=secrets.token_bytes(32)
    return derive_three_way(local_entropy=entropy,future_external_value=b"LOCAL_SELFTEST_ONLY",
        trial_id=trial_id,commit_hash_hex=commit_hash_hex)
