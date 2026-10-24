/**
 * The coding agents with a MemForge memory plugin, and how each one installs
 * it. The plugin gives the agent MCP memory tools: it searches Memories and
 * saves one private Memory when the user asks it to remember something and
 * confirms. It captures nothing on its own and creates no Source, so nothing
 * is configured in the admin UI beyond these commands.
 */

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
