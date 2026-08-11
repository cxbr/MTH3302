from __future__ import annotations

import improved_system_bug_audit_v2 as audit_v2
import verified_improved_benchmark as verified

# Replace the first audit harness with the schema-corrected version before any checks run.
verified.audit = audit_v2


if __name__ == "__main__":
    verified.main()
