# Decision: Kestrel retries a failed job five times

A Kestrel job that raises an error is retried five times, waiting 10, 60, 300,
900 and 3600 seconds between attempts.

After its last retry fails, the job moves to the kestrel-dlq queue, where it
stays until an operator requeues it with `kestrel-admin requeue` or deletes it.
