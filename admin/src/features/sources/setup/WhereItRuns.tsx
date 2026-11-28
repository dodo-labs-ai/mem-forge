import { AlertTriangle, Info, Loader2 } from "lucide-react";
import { useState } from "react";
import { errorMessage } from "@/lib/errors";
import { ChoiceGroup, CopyableCommand, Notice, type Tone } from "@/patterns";
import { Button } from "@/ui/button";
import {
  DAEMON_START_COMMAND,
  DEVICE_SETUP_COMMAND,
  DEVICE_UPGRADE_COMMAND,
  sourceCredentialsCommand,
} from "../model/deviceCommands";
import type { SourceExecutor } from "../model/localAgentSources";
import type { Source, SourceExecutionDevice } from "../model/types";
import { useCredentialCheck } from "../useCredentialCheck";
import type { ChooserDevice, DeviceChoiceStep } from "./model/sourceExecutor";
import { useRemoveDevice, type DeviceChoice } from "./useDeviceChoice";

const EXECUTOR_CHOICES: Readonly<Record<SourceExecutor, { label: string; description: string }>> = {
  server: { label: "Server", description: "The server connects with the credential you enter here." },
  device: { label: "My computer", description: "Your computer connects with the credential stored on it." },
};

const DEVICE_STATES: Readonly<Record<ChooserDevice["state"], { label: string; tone: Tone }>> = {
  online: { label: "Online", tone: "ok" },
  offline: { label: "Offline", tone: "idle" },
  update_required: { label: "Needs an update", tone: "warn" },
};

const DEVICE_STEPS: Readonly<Record<NonNullable<DeviceChoiceStep>, { title: string; detail: string; command: string | null }>> = {
  set_up: {
    title: "Set up MemForge on your computer",
    detail: "Install MemForge on the computer that should run this source, then set up its daemon.",
    command: DEVICE_SETUP_COMMAND,
  },
  choose: {
    title: "Choose the computer that runs this source",
    detail: "Only the chosen computer runs this source.",
    command: null,
  },
  start: {
    title: "Start MemForge on that computer",
    detail: "The chosen computer is offline. Start its daemon, or choose another computer.",
    command: DAEMON_START_COMMAND,
  },
  update: {
    title: "Update MemForge on that computer",
    detail: "The chosen computer needs the server's MemForge version to read this source.",
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
 * My computer's device choice: the chooser's own devices from their
 * reports, with name, version and state, one of which runs the Source (a
 * single device is chosen already), and the step that makes the chosen one
 * ready while it is not. Save then makes the chooser the execution owner
 * and the chosen device the execution device. `moving` says the Source runs
 * on another device now; for a type whose configuration names a folder on
 * the device, that computer's folder becomes the Source.
 */
export function DeviceChoicePanel({ choice, moving, namesDeviceFolder }: { choice: DeviceChoice; moving: boolean; namesDeviceFolder: boolean }) {
  const step = choice.step ? DEVICE_STEPS[choice.step] : null;
  return (
    <section aria-label="Your computers" className="space-y-3 rounded-lg border p-4">
      <p className="text-sm text-muted-foreground">
        {namesDeviceFolder
          ? "The chosen computer runs this source, and its folder picker opens there."
          : "The chosen computer runs this source. After you save, this step shows how to store the credential there."}
      </p>
      {moving && namesDeviceFolder ? (
        <Notice tone="warn" icon={AlertTriangle} title="The new computer's folder becomes the source">
          Files the folder on that computer lacks are removed after its first sync.
        </Notice>
      ) : null}
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
            <div className="space-y-1.5">
              <p id="source-device-label" className="text-sm font-medium">
                Computer
              </p>
              <ChoiceGroup
                id="source-device"
                aria-labelledby="source-device-label"
                choices={choice.devices.map((device) => ({
                  value: device.id,
                  label: device.name,
                  description: `${DEVICE_STATES[device.state].label} · MemForge ${device.version}`,
                }))}
                value={choice.chosen?.id ?? ""}
                onValueChange={choice.choose}
              />
            </div>
          ) : null}
          {step ? (
            <Notice tone="warn" icon={AlertTriangle} title={step.title}>
              <p>{step.detail}</p>
              {step.command ? <CopyableCommand command={step.command} /> : null}
            </Notice>
          ) : null}
          {choice.devices
            .filter((device) => device.state === "offline")
            .map((device) => (
              <OfflineDevice key={device.id} device={device} />
            ))}
        </>
      )}
    </section>
  );
}

/**
 * An offline computer of the person's, which they can remove when they no
 * longer use it. Removing it asks first, because each Source it runs shows
 * Finish setup until they choose another computer.
 */
function OfflineDevice({ device }: { device: ChooserDevice }) {
  const remove = useRemoveDevice();
  const [confirming, setConfirming] = useState(false);
  return (
    <div className="space-y-1.5 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">{device.name} is offline.</span>
        {confirming ? null : (
          <Button type="button" size="xs" variant="ghost" onClick={() => setConfirming(true)}>
            Remove {device.name}
          </Button>
        )}
      </div>
      {confirming ? (
        <div className="flex flex-wrap items-center gap-2">
          <span>Remove {device.name}? Sources it runs show Finish setup until you choose another computer.</span>
          <Button type="button" size="xs" variant="destructive" disabled={remove.isPending} onClick={() => remove.mutate(device.id)}>
            Remove
          </Button>
          <Button type="button" size="xs" variant="outline" onClick={() => setConfirming(false)}>
            Keep
          </Button>
        </div>
      ) : null}
      {remove.error ? (
        <p role="alert" className="text-tone-danger-foreground">
          {errorMessage(remove.error)}
        </p>
      ) : null}
    </div>
  );
}

/** What the execution device's latest report says it lacks, in Configure's words. */
function missingFolders(device: SourceExecutionDevice): string[] {
  return device.not_ready.flatMap((reason) => (reason.kind === "location" && reason.path ? [reason.path] : []));
}

/**
 * A Source that runs on its execution device, as Configure shows it: the
 * device, and what it lacks to run the Source. A missing folder is Finish
 * setup, naming the folder and the device; a credential is stored on the
 * device, a token with `memforge source credentials set <source-id>` or a
 * browser session the Sign in action captures there with a
 * `credential_check` job. Whoever may change where the Source runs
 * (`onMove`) can choose another computer, which opens the device choice:
 * Move while the device is offline, Change computer otherwise.
 */
export function DeviceSetupStep({ source, onMove }: { source: Source; onMove?: () => void }) {
  const signIn = useCredentialCheck();
  const device = source.execution?.device;
  if (!device) return null;
  const owner = source.ownership.viewer_is_execution_owner;
  const ready = source.connection_status?.state === "ready";
  const needsCredential = source.connection_status?.reason === "authentication";
  const folders = missingFolders(device);
  const deviceName = device.name ?? "its owner's computer";

  return (
    <section aria-label="Your computer" className="space-y-3 rounded-lg border p-4">
      {folders.length > 0 ? (
        <Notice tone="warn" icon={AlertTriangle} title="Finish setup">
          {folders.map((folder) => (
            <p key={folder}>
              The folder <span className="font-mono">{folder}</span> is missing on {deviceName}. Create it there, or choose
              another folder.
            </p>
          ))}
        </Notice>
      ) : (
        <Notice
          tone={ready ? "ok" : needsCredential ? "warn" : "idle"}
          icon={needsCredential ? AlertTriangle : Info}
          title={needsCredential ? "Sign in required" : `Runs on ${deviceName}`}
        >
          {ready
            ? "Its credential is stored on that computer."
            : needsCredential
              ? `${deviceName} holds no credential for this source yet. Store it there.`
              : "It syncs once that computer reports it is ready."}
        </Notice>
      )}
      {needsCredential ? (
        !owner ? (
          <p className="text-sm text-muted-foreground">Only the person whose computer runs this source can store its credential.</p>
        ) : device.credential === "secret" ? (
          <div className="space-y-1.5">
            <p className="text-sm text-muted-foreground">Run this on that computer. It asks for the token and keeps it in the computer's keyring.</p>
            <CopyableCommand command={sourceCredentialsCommand(source.id)} />
          </div>
        ) : (
          <div className="space-y-1.5">
            <p className="text-sm text-muted-foreground">Sign in with the browser on that computer. MemForge keeps the session there.</p>
            <Button
              type="button"
              size="sm"
              disabled={signIn.isPending}
              onClick={() => signIn.mutate({ id: source.id, type: source.type, deviceId: device.id })}
            >
              {signIn.isPending ? <Loader2 aria-hidden className="animate-spin" /> : null}
              {signIn.isPending ? "Waiting for sign-in…" : "Sign in"}
            </Button>
            {signIn.error ? (
              <p role="alert" className="text-sm text-tone-danger-foreground">
                {errorMessage(signIn.error)}
              </p>
            ) : null}
          </div>
        )
      ) : null}
      {device.state === "offline" ? (
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-sm text-muted-foreground">{deviceName} is offline.</p>
          {onMove ? (
            <Button type="button" size="sm" variant="outline" onClick={onMove}>
              Move
            </Button>
          ) : null}
        </div>
      ) : onMove ? (
        <Button type="button" size="xs" variant="ghost" onClick={onMove}>
          Change computer
        </Button>
      ) : null}
    </section>
  );
}
