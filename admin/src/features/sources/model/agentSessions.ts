/**
 * Agent-session Sources: the coding agents whose MemForge plugin captures
 * session windows, and how each one installs it. The plugin creates its
 * Source on the first upload, so nothing is configured in the admin UI
 * beyond these commands.
 */

/** The source type the plugins write to. */
export const AGENT_SESSION_SOURCE_TYPE = "agent_session";

/** The GitHub repository that serves the MemForge plugin marketplace. */
const PLUGIN_MARKETPLACE = "dodo-labs-ai/mem-forge";
/** The plugin, as `<plugin>@<marketplace>`. */
const PLUGIN = "memory@memforge";

export interface AgentPluginSetup {
  /** The client id the plugin reports, which picks the brand mark: "codex" or "claude-code". */
  client: string;
  name: string;
  /** Where the commands run. */
  runFrom: string;
  /** The install commands, in order, as the plugin README gives them. */
  commands: readonly string[];
}

export const AGENT_PLUGIN_SETUPS: readonly AgentPluginSetup[] = [
  {
    client: "codex",
    name: "Codex",
    runFrom: "Run these in a terminal where the codex command is available.",
    commands: [`codex plugin marketplace add ${PLUGIN_MARKETPLACE}`, `codex plugin add ${PLUGIN}`],
  },
  {
    client: "claude-code",
    name: "Claude Code",
    runFrom: "Run these inside a Claude Code session.",
    commands: [`/plugin marketplace add ${PLUGIN_MARKETPLACE}`, `/plugin install ${PLUGIN}`],
  },
];
