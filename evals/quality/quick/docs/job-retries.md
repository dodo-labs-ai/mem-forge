# Decision: Kestrel retries a failed job three times

A Kestrel job that raises an error is retried three times, waiting 10, 60 and
300 seconds between attempts. After the third failure the job moves to the
kestrel-dlq queue, where it stays until someone requeues or deletes it. We
chose a short, fixed schedule over exponential backoff so that an operator can
tell from the clock alone when a job will run again.
