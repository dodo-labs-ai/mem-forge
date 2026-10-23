import { Plus } from "lucide-react";
import { lazy, Suspense, useState } from "react";
import { isReservedProjectKey, useProjectListQuery } from "@/api";
import { ChoiceGroup, FormField } from "@/patterns";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { Switch } from "@/ui/switch";
import {
  ACCESS_CHOICES,
  scheduleIntervalChoices,
  scheduleIntervalLabel,
  type AccessPolicy,
  type SyncSchedule,
} from "./model/productSteps";

// Loaded when it opens, so the Sources pages and the Projects feature do not import each other.
const CreateProjectDialog = lazy(() =>
  import("@/features/projects").then((module) => ({ default: module.CreateProjectDialog })),
);

export function BasicsStep({ name, onNameChange, error }: { name: string; onNameChange: (name: string) => void; error?: string }) {
  return (
    <FormField
      label="Source name"
      htmlFor="source-name"
      hint="Use a name your workspace members will recognize."
      hintId="source-name-hint"
      error={error}
      errorId="source-name-error"
    >
      <Input
        id="source-name"
        value={name}
        autoComplete="off"
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? "source-name-error source-name-hint" : "source-name-hint"}
        onChange={(event) => onNameChange(event.target.value)}
      />
    </FormField>
  );
}

export function AccessStep({
  value,
  onChange,
  error,
}: {
  value: AccessPolicy | null;
  onChange: (value: AccessPolicy) => void;
  error?: string;
}) {
  return (
    <div className="space-y-1.5">
      <p id="source-access-label" className="text-sm font-medium">
        Who can use this source?
      </p>
      <ChoiceGroup
        id="source-access"
        aria-labelledby="source-access-label"
        aria-describedby={error ? "source-access-error" : undefined}
        aria-invalid={Boolean(error)}
        choices={ACCESS_CHOICES.map((choice) => ({ value: choice.value, label: choice.label, description: choice.description }))}
        value={value ?? ""}
        onValueChange={(next) => onChange(next as AccessPolicy)}
      />
      {error ? (
        <p id="source-access-error" className="text-xs text-tone-danger-foreground">
          {error}
        </p>
      ) : null}
      <p className="text-xs text-muted-foreground">You can change this later from the source's details.</p>
    </div>
  );
}

const NO_PROJECT = "";

/**
 * The project the Source's memories go to, chosen from the workspace's own
 * projects or created here. No project leaves the Source to be mapped later,
 * and it asks for a project until then.
 */
export function ProjectStep({
  projectKey,
  onChange,
  routedByField,
}: {
  projectKey: string | null;
  onChange: (projectKey: string | null) => void;
  /** The Source routes memories by a configuration field, which this step does not change. */
  routedByField: boolean;
}) {
  const projectList = useProjectListQuery();
  const [createOpen, setCreateOpen] = useState(false);
  if (routedByField) {
    return <p className="text-sm text-muted-foreground">This source sends each memory to the project its repository maps to.</p>;
  }
  const projects = (projectList.data?.data ?? []).filter((project) => !isReservedProjectKey(project.key));
  const names = new Map(projects.map((project) => [project.key, project.name]));
  const canCreate = projectList.data?.can_manage ?? false;

  return (
    <div className="space-y-3">
      <FormField label="Save memories to" htmlFor="source-project" hint="Memories from this source go to this project." hintId="source-project-hint">
        <Select
          value={projectKey ?? NO_PROJECT}
          onValueChange={(next) => {
            if (typeof next === "string") onChange(next === NO_PROJECT ? null : next);
          }}
        >
          <SelectTrigger id="source-project" aria-describedby="source-project-hint" className="w-full sm:w-80">
            <SelectValue>
              {(value: string) => (value === NO_PROJECT ? "No project yet" : (names.get(value) ?? value))}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NO_PROJECT}>No project yet</SelectItem>
            {projects.map((project) => (
              <SelectItem key={project.key} value={project.key}>
                {project.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </FormField>
      {canCreate ? (
        <Button type="button" variant="outline" size="sm" onClick={() => setCreateOpen(true)}>
          <Plus />
          New project
        </Button>
      ) : null}
      {createOpen ? (
        <Suspense fallback={null}>
          <CreateProjectDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={(project) => onChange(project.key)} />
        </Suspense>
      ) : null}
    </div>
  );
}

export function ScheduleStep({ schedule, onChange }: { schedule: SyncSchedule; onChange: (schedule: SyncSchedule) => void }) {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3">
        <Switch
          id="source-schedule-enabled"
          aria-describedby="source-schedule-hint"
          checked={schedule.enabled}
          onCheckedChange={(enabled) => onChange({ ...schedule, enabled })}
        />
        <div className="space-y-0.5">
          <label htmlFor="source-schedule-enabled" className="text-sm font-medium">
            Sync automatically
          </label>
          <p id="source-schedule-hint" className="text-xs text-muted-foreground">
            You can still sync by hand at any time.
          </p>
        </div>
      </div>
      {schedule.enabled ? (
        <FormField label="Frequency" htmlFor="source-schedule-interval">
          <Select
            value={String(schedule.interval_minutes)}
            onValueChange={(next) => {
              if (typeof next === "string") onChange({ ...schedule, interval_minutes: Number(next) });
            }}
          >
            <SelectTrigger id="source-schedule-interval" className="w-full sm:w-60">
              <SelectValue>{(value: string) => scheduleIntervalLabel(Number(value))}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {scheduleIntervalChoices(schedule.interval_minutes).map((interval) => (
                <SelectItem key={interval.minutes} value={String(interval.minutes)}>
                  {interval.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FormField>
      ) : null}
    </div>
  );
}
