# Kestrel queue limits

- A job payload may be at most 1 MiB. A larger payload is rejected with the
  error `KES-PAYLOAD-TOO-LARGE`.
- Each tenant may have at most 500 jobs in flight; further jobs wait in the
  tenant's backlog.
