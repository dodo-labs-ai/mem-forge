import { Copy } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { AGENT_PLUGIN_SETUPS, AGENT_SESSION_SOURCE_TYPE, type AgentPluginSetup } from "../model/agentSessions";
import { SourceIcon } from "../SourceIcon";

function CopyableCommand({ command }: { command: string }) {
  function copy() {
    navigator.clipboard.writeText(command).then(
      () => toast.success("Command copied"),
      () => toast.error("Could not copy"),
    );
  }
  return (
    <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-2">
      <code className="min-w-0 flex-1 font-mono text-xs break-all">{command}</code>
      <Button type="button" size="sm" variant="outline" aria-label={`Copy ${command}`} onClick={copy}>
        <Copy aria-hidden />
        Copy
      </Button>
    </div>
  );
}

function PluginSetupDialog({
  setup,
  open,
  onOpenChange,
}: {
  setup: AgentPluginSetup;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Set up {setup.name}</DialogTitle>
          <DialogDescription>
            The MemForge plugin runs on your computer, records your coding sessions and sends redacted session windows
            to MemForge as you work. Its source appears on the Sources page after the first upload; nothing else is set
            up here.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">{setup.runFrom}</p>
            {setup.commands.map((command) => (
              <CopyableCommand key={command} command={command} />
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            To reach a MemForge other than the default, set <code className="font-mono">MEMFORGE_API_URL</code>, and{" "}
            <code className="font-mono">MEMFORGE_API_TOKEN</code> when that server asks for a token. The plugin README
            explains how it chooses a workspace.
          </p>
        </div>
        <DialogFooter showCloseButton />
      </DialogContent>
    </Dialog>
  );
}

/**
 * The type picker's "Coding agent integrations": Codex and Claude Code, each
 * opening the commands that install its MemForge plugin. Their Sources are
 * created by the plugin, so the picker offers no setup page for them.
 */
export function AgentPluginSetupSection() {
  const [selected, setSelected] = useState<AgentPluginSetup>(AGENT_PLUGIN_SETUPS[0]!);
  const [open, setOpen] = useState(false);
  return (
    <section aria-labelledby="agent-integrations-heading" className="space-y-3">
      <div className="space-y-1">
        <h2 id="agent-integrations-heading" className="text-base font-medium">
          Coding agent integrations
        </h2>
        <p className="text-sm text-muted-foreground">Save what your coding agent learns while it works as memories.</p>
      </div>
      <ul aria-label="Coding agents" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {AGENT_PLUGIN_SETUPS.map((setup) => (
          <li key={setup.client}>
            <button
              type="button"
              className="flex h-full w-full flex-col gap-2 rounded-lg border p-4 text-left outline-none transition-colors hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50"
              onClick={() => {
                setSelected(setup);
                setOpen(true);
              }}
            >
              <span className="flex items-center gap-2.5">
                <SourceIcon type={AGENT_SESSION_SOURCE_TYPE} client={setup.client} className="size-5" />
                <span className="font-medium text-foreground">{setup.name}</span>
              </span>
              <span className="text-sm text-muted-foreground">Install the MemForge plugin to save {setup.name} sessions as memories.</span>
            </button>
          </li>
        ))}
      </ul>
      <PluginSetupDialog setup={selected} open={open} onOpenChange={setOpen} />
    </section>
  );
}
