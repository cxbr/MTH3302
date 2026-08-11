from __future__ import annotations

import improved_system_bug_audit as audit_core
import improved_system_bug_audit_v7 as audit_v7
import verified_improved_benchmark as verified

# The final audit module owns the pass/fail gate and output paths. The inclusive
# MFE/MAE diagnostic correction is defined in the original audit utility.
audit_v7.corrected_diagnostics = audit_core.corrected_diagnostics
verified.audit = audit_v7


if __name__ == "__main__":
    verified.main()
