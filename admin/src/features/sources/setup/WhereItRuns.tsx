import { AlertTriangle, Info, Loader2 } from "lucide-react";
import { errorMessage } from "@/lib/errors";
import { ChoiceGroup, CopyableCommand, Notice, StatusBadge, type Tone } from "@/patterns";
import { Button } from "@/ui/button";
import {
  DAEMON_START_COMMAND,
  DEVICE_SETUP_COMMAND,
  DEVICE_UPGRADE_COMMAND,
  sourceCredentialsCommand,
} from "../model/deviceCommands";
import type { SourceExecutor } from "../model/localAgentSources";
import type { Source } from "../model/types";
import { useCredentialCheck } from "../useCredentialCheck";
import type { ChooserDevice, DeviceChoiceStep } from "./model/sourceExecutor";
import type { DeviceChoice } from "./useDeviceChoice";

const EXECUTOR_CHOICES: Readonly<Record<SourceExecutor, { label: string; description: string }>> = {
  server: { label: "Server", description: "The server connects with the credential you enter here." },
  device: { label: "My computer", description: "Your computer connects with the credential stored on it." },
};

const DEVICE_STATES: Readonly<Record<ChooserDevice["state"], { label: string; tone: Tone }>> = {
  online: { label: "Online", tone: "ok" },
  offline: { label: "Offline", tone: "idle" },
  update_required: { label: "Needs an update", tone: "warn" },
};

const DEVICE_STEPS: Readonly<Record<NonNullable<DeviceChoiceStep>, { title: string; detail: string; command: string }>> = {
  set_up: {
    title: "Set up MemForge on your computer",
    detail: "Install MemForge on the computer that should run this source, then set up its daemon.",
    command: DEVICE_SETUP_COMMAND,
  },
  start: {
    title: "Start MemForge on your computer",
    detail: "None of your computers is online. Start the daemon on the one that should run this source.",
    command: DAEMON_START_COMMAND,
  },
  update: {
    title: "Update MemForge on your computer",
    detail: "Your computer needs the server's MemForge version to read this source.",
    command: DEVICE_UPGRADE_COMMAND,
  },
};

/** Configure's Where it runs: Server or My computer, from the executors the type declares. */
export function WhereItRunsChoice({
  executors,
  value,
  onChange,
}: {
  executors: readonly SourceExecutor[];
  value: SourceExecutor;
  onChange: (executor: SourceExecutor) => void;
}) {
  return (
    <div className="space-y-1.5">
      <p id="source-executor-label" className="text-sm font-medium">
        Where it runs
      </p>
      <ChoiceGroup
        id="source-executor"
        aria-labelledby="source-executor-label"
        choices={executors.map((executor) => ({ value: executor, ...EXECUTOR_CHOICES[executor] }))}
        value={value}
        onValueChange={(next) => onChange(next as SourceExecutor)}
      />
    </div>
  );
}

/**
 * My computer chosen for a Source that does not run on a device yet: the
 * chooser's own devices, for information, and the step that makes one
 * ready while none is. Save then moves the Source and makes the chooser its
 * execution owner; the credential is stored on the computer afterwards.
 */
export function DeviceChoicePanel({ choice }: { choice: DeviceChoice }) {
  const step = choice.step ? DEVICE_STEPS[choice.step] : null;
  return (
    <section aria-label="Your computers" className="space-y-3 rounded-lg border p-4">
      <p className="text-sm text-muted-foreground">
        The computer where this source's credential is stored runs it. After you save, this step shows how to store the
        credential there.
      </p>
      {choice.pending ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 aria-hidden className="size-4 animate-spin" />
          Checking your computers…
        </p>
      ) : choice.error ? (
        <p role="alert" className="text-sm text-tone-danger-foreground">
          Could not read your computers: {errorMessage(choice.error)}
        </p>
      ) : (
        <>
          {choice.devices.length > 0 ? (
            <ul aria-label="Your computers" className="divide-y rounded-md border">
              {choice.devices.map((device) => (
                <li key={device.id} className="flex items-center justify-between gap-2 px-3 py-2 text-sm">
                  <span className="min-w-0 truncate">
                    {device.name} <span className="text-muted-foreground">MemForge {device.version}</span>
                  </span>
                  <StatusBadge tone={DEVICE_STATES[device.state].tone}>{DEVICE_STATES[device.state].label}</StatusBadge>
                </li>
              ))}
            </ul>
          ) : null}
          {step ? (
            <Notice tone="warn" icon={AlertTriangle} title={step.title}>
              <p>{step.detail}</p>
              <CopyableCommand command={step.command} />
            </Notice>
          ) : null}
        </>
      )}
    </section>
  );
}

/**
 * The credential of a Source that runs on its execution owner's device,
 * which stays on that device: a token stored with
 * `memforge source credentials set <source-id>`, or a browser session the
 * Sign in action captures there with a `credential_check` job. Until a
 * device report names the Source, its connection asks for sign-in.
 */
export function DeviceCredentialStep({ source }: { source: Source }) {
  const signIn = useCredentialCheck();
  const device = source.execution?.device;
  const signedIn = source.connection_status?.state === "ready";
  if (!device) return null;

  return (
    <section aria-label="Credential on your computer" className="space-y-3 rounded-lg border p-4">
      <Notice tone={signedIn ? "ok" : "warn"} icon={signedIn ? Info : AlertTriangle} title={signedIn ? `Runs on ${device.name ?? "its owner's computer"}` : "Sign in required"}>
        {signedIn
          ? "Its credential is stored on that computer."
          : "No computer holds this source's credential yet. Store it on the computer that should run the source."}
      </Notice>
      {!source.ownership.viewer_is_execution_owner ? (
        <p className="text-sm text-muted-foreground">Only the person whose computer runs this source can store its credential.</p>
      ) : device.credential === "secret" ? (
        <div className="space-y-1.5">
          <p className="text-sm text-muted-foreground">Run this on that computer. It asks for the token and keeps it in the computer's keyring.</p>
          <CopyableCommand command={sourceCredentialsCommand(source.id)} />
        </div>
      ) : (
        <div className="space-y-1.5">
          <p className="text-sm text-muted-foreground">Sign in with the browser on that computer. MemForge keeps the session there.</p>
          <Button type="button" size="sm" disabled={signIn.isPending} onClick={() => signIn.mutate({ id: source.id, type: source.type })}>
            {signIn.isPending ? <Loader2 aria-hidden className="animate-spin" /> : null}
            {signIn.isPending ? "Waiting for sign-in…" : "Sign in"}
          </Button>
          {signIn.error ? (
            <p role="alert" className="text-sm text-tone-danger-foreground">
              {errorMessage(signIn.error)}
            </p>
          ) : null}
        </div>
      )}
    </section>
  );
}
