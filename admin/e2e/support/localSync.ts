const ACQUISITION_CONTRACTS = { confluence: 2, jira: 2, notion: 2, github_pages: 2, github_repo: 2, local_markdown: 3 };

/** The viewer's local sync status: one device on the server's contracts when online, none when offline. */
export function localSyncStatus(online: boolean) {
  const devices = online
    ? [{ device_id: "dev-1", name: "Alex's MacBook", version: "0.2.0b7", status: "online", last_seen_at: "2026-11-27T10:00:00Z", acquisition_contracts: ACQUISITION_CONTRACTS }]
    : [];
  return {
    status: online ? "online" : "offline",
    last_seen_at: online ? "2026-11-27T10:00:00Z" : null,
    checked_at: "2026-11-27T10:00:00Z",
    stale_after_seconds: 90,
    devices,
    sources: [],
    server: { version: "0.2.0b7", acquisition_contracts: ACQUISITION_CONTRACTS },
  };
}
