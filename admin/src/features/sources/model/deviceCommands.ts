/** The commands a person runs on their computer to make it collect a Source. */

/** Installs MemForge, names this computer and installs its daemon. */
export const DEVICE_SETUP_COMMAND = "memforge setup";

/** Starts the installed daemon. */
export const DAEMON_START_COMMAND = "memforge daemon start";

/** Installs exactly the server's MemForge version and restarts the daemon. */
export const DEVICE_UPGRADE_COMMAND = "memforge upgrade";

/** Stores a device Source's token in this computer's keyring. */
export function sourceCredentialsCommand(sourceId: string): string {
  return `memforge source credentials set ${sourceId}`;
}
