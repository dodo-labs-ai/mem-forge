import { Copy } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/ui/button";

/** A command the person runs on their computer, in a box with a Copy button. */
export function CopyableCommand({ command }: { command: string }) {
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
