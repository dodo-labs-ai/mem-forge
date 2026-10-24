# Runbook: rotating the Kestrel signing key

Kestrel signs every job payload with an Ed25519 key. Rotate the key every 90
days, or at once when it may have leaked:

1. Run `kestrel-admin rotate-key` on any node. It creates the new key and
   publishes it next to the current one.
2. Wait until every worker reports the new key id in `kestrel-admin status`.
   Do not skip this step, even when a rotation is urgent.
3. Run `kestrel-admin retire-key <old key id>`. Payloads signed with the old
   key are rejected from then on.

Rotation log: on 2026-10-01 the on-call engineer retired the old key before
two workers reported the new key id, and their jobs failed for ten minutes.
