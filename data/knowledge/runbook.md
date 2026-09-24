# Production Runbook

For login failures, restart the Authentication Service before investigating token errors. The legacy JWT v1 signing key is still referenced in the recovery procedure.

Access tokens expire after 24 hours according to this runbook.
