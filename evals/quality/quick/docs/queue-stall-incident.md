# Incident review: the Kestrel queue stalled on 2026-09-12

For 40 minutes no job was picked up. The cause was a configuration change that
turned fsync off on the queue database, so a crash left the WAL file corrupt
and every worker waited on a lock that was never released. We restored fsync,
added a startup check that refuses to run with fsync off, and now page the
on-call engineer when no job has started for five minutes.
