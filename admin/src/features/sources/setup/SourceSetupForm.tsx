import { AlertTriangle, Info, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate, useSearchParams, type Blocker } from "react-router-dom";
import { toast } from "sonner";
import { useProjects, useSharedWorkspace, type components } from "@/api";
import { errorMessage } from "@/lib/errors";
import { SOURCE_SETTINGS_STEP_PARAM, SOURCES_PATH } from "@/lib/paths";
import { Notice, PropertyList, StepRail, useUnsavedChangesGuard, type Property, type RailStep } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { sourceExecutor, type SourceExecutor } from "../model/localAgentSources";
import { sourceProjectBinding } from "../model/projectBinding";
import { ACCESS_CHOICES, type AccessPolicy } from "../model/sourceAccess";
import { scopeChangeErrorMessage } from "../model/sourceScopeTransition";
import type { Source } from "../model/types";
import {
  useChangeSourceExecutor,
  useConfigCheck,
  useCreateSource,
  useCheckSetup,
  useDebouncedScopeChangePreview,
  useUpdateSource,
  type CreateSourceBody,
  type UpdateSourceBody,
} from "./api";
import { CheckSetupPanel } from "./CheckSetupPanel";
import { ConfigGroupFields } from "./ConfigGroupFields";
import { checkSetupKind } from "./model/checkSetup";
import {
  comparisonString,
  defaultValues,
  formSchema,
  missingRequired,
  reconcileHiddenChoices,
  serializeConfig,
  type ConfigFormState,
  type ConfigSchema,
  type ConfigValues,
  type FieldValue,
  type ServerConfig,
} from "./model/configForm";
import {
  fixedProjectBinding,
  initialSchedule,
  PERSONAL_ACCESS_POLICY,
  scheduleSummary,
  type SyncSchedule,
} from "./model/productSteps";
import { activeSyncRunOf, runningSyncOfRefusedSave } from "./model/runningSyncSave";
import { scopeChangeNotice } from "./model/scopeChangeNotice";
import { deviceFormSchema, executorChoices } from "./model/sourceExecutor";
import { configIssuesOf, issueView, type FieldIssue } from "./model/setupIssues";
import { checkSetupGroup, groupComplete, groupSummary, issueCountsByGroup, setupSteps, type SetupStep } from "./model/setupSteps";
import { AccessStep, BasicsStep, ProjectStep, ScheduleStep } from "./ProductSteps";
import { useRunningSyncSave } from "./useRunningSyncSave";
import { useDeviceChoice } from "./useDeviceChoice";
import { DeviceChoicePanel, DeviceCredentialStep, WhereItRunsChoice } from "./WhereItRuns";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];
type SavePayload = CreateSourceBody | UpdateSourceBody;

const DERIVED_NOTE = "Filled from the URL";
const NAME_REQUIRED = "Name this source.";
const ACCESS_REQUIRED = "Choose who can use this source.";

interface SourceSetupFormProps {
  gene: GeneMetadata;
  schema: ConfigSchema;
  /** The Source being edited; absent while one is created. */
  source?: Source;
}

/** The field issues the form itself finds before save: a visible required field without a value, a malformed number. */
function clientFieldIssues(schema: ConfigSchema, state: ConfigFormState): Record<string, string> {
  const parsed = formSchema(schema, { derived: state.derived, stored: state.stored }).safeParse(state.values);
  if (parsed.success) return {};
  const issues: Record<string, string> = {};
  for (const issue of parsed.error.issues) {
    const key = issue.path[0];
    if (typeof key === "string" && !issues[key]) issues[key] = issue.message;
  }
  return issues;
}

function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Creates a Source of one type, or edits one, drawn from the type's form
 * description: Basics, the type's groups, Access, Project and Schedule, one
 * step at a time, with `?step=` naming the open step.
 *
 * - config-check runs when a field loses focus, when an edit page opens, and
 *   before save. A derived value fills only a field nobody set; issues map
 *   onto fields, shown once a field was left or a save was tried.
 * - On an edit page, changing a scope field previews what saving would do
 *   to the scope; a refused change blocks save, and save sends the scope
 *   change the page saw in progress.
 * - A save while a sync runs asks to stop it first or wait.
 * - Leaving with unsaved changes asks first.
 * - A type that runs on the server or a device shows Where it runs on its
 *   connection step. On My computer the form holds no secret: a new choice
 *   saves only while one of the person's devices is ready, and a Source
 *   that runs on a device shows how to store its credential there. Save
 *   changes the executor first, then the rest of the settings.
 */
export function SourceSetupForm({ gene, schema: typeSchema, source }: SourceSetupFormProps) {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const sharedWorkspace = useSharedWorkspace();
  const projects = useProjects();
  const sourceId = source?.id;
  const stored = (source?.config ?? {}) as ServerConfig;
  const canEditConfig = !source || Boolean(source.capabilities?.can_configure_connection);
  const binding = source ? sourceProjectBinding(source) : null;
  const routedByField = binding?.mode === "by_field";
  const storedExecutor: SourceExecutor = source ? sourceExecutor(source) : "server";
  const [executor, setExecutor] = useState<SourceExecutor>(storedExecutor);
  const executorOptions = canEditConfig ? executorChoices(gene.executors) : [];
  const executorChanged = executor !== storedExecutor;
  const schema = useMemo(() => (executor === "device" ? deviceFormSchema(typeSchema) : typeSchema), [executor, typeSchema]);
  const choosesDevice = executor === "device" && executorChanged;
  const deviceChoice = useDeviceChoice(gene.name, choosesDevice);
  const awaitsDevice = choosesDevice && !deviceChoice.ready;

  const [initial] = useState(() => {
    const values = defaultValues(typeSchema, source?.config ?? {});
    return {
      name: source?.name ?? "",
      values,
      config: serializeConfig(typeSchema, values),
      projectKey: binding?.mode === "fixed" ? (binding.project_key ?? null) : null,
      schedule: initialSchedule(source, gene),
    };
  });
  const [name, setName] = useState(initial.name);
  const [values, setValues] = useState<ConfigValues>(initial.values);
  const valuesRef = useRef(values);
  const [access, setAccess] = useState<AccessPolicy | null>(null);
  const [projectKey, setProjectKey] = useState(initial.projectKey);
  const [schedule, setSchedule] = useState<SyncSchedule>(initial.schedule);
  const [touched, setTouched] = useState<ReadonlySet<string>>(() => new Set());
  const [editedSinceCheck, setEditedSinceCheck] = useState<ReadonlySet<string>>(() => new Set());
  const [submitted, setSubmitted] = useState(false);
  const [refusalIssues, setRefusalIssues] = useState<FieldIssue[] | null>(null);
  const [saveError, setSaveError] = useState<unknown>(null);
  const [checkingBeforeSave, setCheckingBeforeSave] = useState(false);
  const [saved, setSaved] = useState(false);

  const configCheck = useConfigCheck(gene.name, sourceId, executor);
  const createSource = useCreateSource();
  const updateSource = useUpdateSource(sourceId ?? "");
  const changeExecutor = useChangeSourceExecutor(sourceId ?? "");
  const state: ConfigFormState = { values, derived: configCheck.result?.derived, stored };
  // The server reads no provider for a Source that runs on a device.
  const previewKind = executor === "device" ? null : checkSetupKind(gene.name, schema, state);
  const discovery = useCheckSetup(gene.name, previewKind, executor, source);
  const config = serializeConfig(schema, values);
  const steps = setupSteps(schema, { mode: source ? "edit" : "create", sharedWorkspace });
  const currentKey = steps.find((step) => step.key === searchParams.get(SOURCE_SETTINGS_STEP_PARAM))?.key ?? steps[0]!.key;
  const currentIndex = steps.findIndex((step) => step.key === currentKey);
  const current = steps[currentIndex]!;
  const factsGroup = checkSetupGroup(schema) ?? steps.findLast((step) => step.kind === "group")?.key;

  const { check } = configCheck;
  useEffect(() => {
    // An edit page checks the stored configuration once, so a blank mode shows its effective value.
    if (sourceId && canEditConfig) void check(initial.config).catch(() => undefined);
  }, [check, sourceId, canEditConfig, initial.config]);

  // What the person changed that a save would send.
  const settingsDirty =
    name !== initial.name ||
    !sameJson(config, initial.config) ||
    projectKey !== initial.projectKey ||
    !sameJson(schedule, initial.schedule) ||
    access !== null;
  const dirty = settingsDirty || executorChanged;
  const blocker = useUnsavedChangesGuard(dirty && !saved);

  // Declared after the guard, so the guard has let go of the saved page before it leaves.
  useEffect(() => {
    if (!saved) return;
    toast.success(source ? `${name.trim()} saved` : `${name.trim()} created`);
    navigate(SOURCES_PATH);
  }, [saved, source, name, navigate]);

  const secretKeys = useMemo(
    () => typeSchema.fields.filter((field) => field.field_type === "secret").map((field) => field.key),
    [typeSchema],
  );
  const scopeChanged =
    source !== undefined && canEditConfig && schema.fields.some((field) => field.scope && !sameJson(config[field.key], initial.config[field.key]));
  const missing = missingRequired(schema, state);
  const scopePreview = useDebouncedScopeChangePreview(sourceId ?? "", scopeChanged && missing.length === 0 ? config : null, secretKeys);
  const scopeNotice = scopeChanged ? scopeChangeNotice(scopePreview) : null;
  const scopeRefused = scopeNotice?.tone === "danger";

  /** The secrets the person typed, which a move to the server sends with the executor change. */
  function typedSecrets(): Record<string, string> {
    return Object.fromEntries(
      secretKeys.flatMap((key) => {
        const value = valuesRef.current[key];
        return typeof value === "string" && value.trim() !== "" ? [[key, value]] : [];
      }),
    );
  }

  /**
   * Saves an edited Source: a changed executor first, with the secrets a move
   * to the server needs, then the other settings, whose configuration then
   * leaves the secrets out, as the executor change already holds them.
   */
  async function saveSource(payload: UpdateSourceBody) {
    if (!executorChanged) {
      await updateSource.mutateAsync(payload);
      return;
    }
    await changeExecutor.mutateAsync(executor === "server" ? { executor, secrets: typedSecrets() } : { executor });
    if (!settingsDirty) return;
    const config = payload.config
      ? Object.fromEntries(Object.entries(payload.config).filter(([key]) => !secretKeys.includes(key)))
      : undefined;
    await updateSource.mutateAsync({ ...payload, ...(config ? { config } : {}) });
  }

  async function persist(payload: SavePayload) {
    setSaveError(null);
    try {
      if (source) await saveSource(payload as UpdateSourceBody);
      else await createSource.mutateAsync(payload as CreateSourceBody);
    } catch (error) {
      if (runningSyncOfRefusedSave(error) === null) {
        const issues = configIssuesOf(error);
        if (issues) {
          setRefusalIssues(issues);
          openFirstStepWithIssue(issues);
        } else {
          setSaveError(error);
        }
      }
      throw error;
    }
    setSaved(true);
  }
  const runningSync = useRunningSyncSave<SavePayload>({ sourceId, activeRunId: activeSyncRunOf(source), save: persist });

  // Issues as shown: the form's own once a save was tried, and the server's for fields left since its last check.
  const serverIssues = issueView(refusalIssues ?? configCheck.result?.issues ?? []);
  const shownFieldIssues: Record<string, string> = {};
  for (const [key, message] of Object.entries(serverIssues.byField)) {
    if ((submitted || touched.has(key)) && !editedSinceCheck.has(key)) shownFieldIssues[key] = message;
  }
  if (submitted && canEditConfig) Object.assign(shownFieldIssues, clientFieldIssues(schema, state));
  const generalIssues = submitted || touched.size > 0 ? serverIssues.general : [];
  const issueCounts = issueCountsByGroup(schema, Object.keys(shownFieldIssues));
  const nameError = submitted && name.trim() === "" ? NAME_REQUIRED : undefined;
  const accessError = submitted && access === null ? ACCESS_REQUIRED : undefined;

  // A derived value is noted once it answers what the person typed, not the stored configuration an edit page opened with.
  const notes: Record<string, string> = {};
  for (const field of source && touched.size === 0 ? [] : schema.fields) {
    const derived = configCheck.result?.derived[field.key];
    if (values[field.key] === undefined && comparisonString(derived, field.field_type) !== "") notes[field.key] = DERIVED_NOTE;
  }

  function openStep(key: string) {
    setSearchParams(
      (params) => {
        const next = new URLSearchParams(params);
        next.set(SOURCE_SETTINGS_STEP_PARAM, key);
        return next;
      },
      { preventScrollReset: true },
    );
  }

  function openFirstStepWithIssue(issues: readonly FieldIssue[]) {
    const counts = issueCountsByGroup(schema, issues.map((issue) => issue.key));
    const step = steps.find((candidate) => candidate.kind === "group" && (counts[candidate.key] ?? 0) > 0);
    if (step) openStep(step.key);
  }

  function changeField(key: string, value: FieldValue) {
    const next = reconcileHiddenChoices(schema, { ...state, values: { ...valuesRef.current, [key]: value } });
    valuesRef.current = next;
    setValues(next);
    setEditedSinceCheck((keys) => new Set(keys).add(key));
  }

  function checkConfig() {
    setRefusalIssues(null);
    setEditedSinceCheck(new Set());
    return check(serializeConfig(schema, valuesRef.current));
  }

  function leaveField(key: string) {
    setTouched((keys) => new Set(keys).add(key));
    if (canEditConfig) void checkConfig().catch(() => undefined);
  }

  function checkSetup() {
    discovery.mutate(config, {
      onError: (error) => {
        const issues = configIssuesOf(error);
        if (!issues) return;
        setRefusalIssues(issues);
        setTouched((keys) => new Set([...keys, ...issues.flatMap((issue) => (issue.key ? [issue.key] : []))]));
      },
    });
  }

  function savePayload(): SavePayload {
    const common = {
      name: name.trim(),
      sync_schedule: schedule,
      ...(routedByField ? {} : { project_binding: fixedProjectBinding(projectKey) }),
    };
    if (!source) {
      return { ...common, type: gene.name, access_policy: sharedWorkspace ? access! : PERSONAL_ACCESS_POLICY, config, executor };
    }
    return {
      ...common,
      ...(canEditConfig ? { config } : {}),
      ...(source.scope_transition ? { expected_scope_transition_id: source.scope_transition.id } : {}),
    };
  }

  async function submit() {
    setSubmitted(true);
    setSaveError(null);
    const fieldIssues = canEditConfig ? clientFieldIssues(schema, state) : {};
    const fieldCounts = issueCountsByGroup(schema, Object.keys(fieldIssues));
    const blocked = steps.find(
      (step) =>
        (step.kind === "basics" && name.trim() === "") ||
        (step.kind === "group" && (fieldCounts[step.key] ?? 0) > 0) ||
        (step.kind === "access" && access === null),
    );
    if (blocked) {
      openStep(blocked.key);
      return;
    }
    if (canEditConfig) {
      setCheckingBeforeSave(true);
      try {
        const answer = await checkConfig();
        if (answer.issues.length > 0) {
          openFirstStepWithIssue(answer.issues);
          return;
        }
      } catch (error) {
        setSaveError(error);
        return;
      } finally {
        setCheckingBeforeSave(false);
      }
    }
    runningSync.save(savePayload());
  }

  const projectNames = new Map((projects.data ?? []).map((project) => [project.key, project.name]));
  const railSteps: RailStep[] = steps.map((step) => {
    switch (step.kind) {
      case "basics":
        return { key: step.key, label: step.label, complete: name.trim() !== "", errorCount: nameError ? 1 : 0, summary: name.trim() };
      case "group":
        return {
          key: step.key,
          label: step.label,
          complete: !canEditConfig || groupComplete(schema, step.key, state),
          errorCount: issueCounts[step.key] ?? 0,
          summary: canEditConfig ? groupSummary(schema, step.key, state) : "Set by the source owner",
        };
      case "access":
        return {
          key: step.key,
          label: step.label,
          complete: access !== null,
          errorCount: accessError ? 1 : 0,
          summary: ACCESS_CHOICES.find((choice) => choice.value === access)?.label,
        };
      case "project":
        return {
          key: step.key,
          label: step.label,
          complete: true,
          summary: routedByField ? "By repository" : projectKey ? (projectNames.get(projectKey) ?? projectKey) : "No project yet",
        };
      case "schedule":
        return { key: step.key, label: step.label, complete: true, summary: scheduleSummary(schedule) };
    }
  });

  const saving =
    checkingBeforeSave || createSource.isPending || updateSource.isPending || changeExecutor.isPending || runningSync.stopping;
  const saveLabel = source ? "Save changes" : "Create source";
  const previous = steps[currentIndex - 1];
  const next = steps[currentIndex + 1];

  function stepContent(step: SetupStep) {
    switch (step.kind) {
      case "basics":
        return <BasicsStep name={name} onNameChange={setName} error={nameError} />;
      case "access":
        return <AccessStep value={access} onChange={setAccess} error={accessError} />;
      case "project":
        return <ProjectStep projectKey={projectKey} onChange={setProjectKey} routedByField={routedByField} />;
      case "schedule":
        return <ScheduleStep schedule={schedule} onChange={setSchedule} />;
      case "group":
        return canEditConfig ? (
          <GroupStep
            step={step}
            schema={schema}
            state={state}
            sourceType={gene.name}
            sourceId={sourceId}
            issues={shownFieldIssues}
            notes={notes}
            facts={groupFacts(step)}
            executor={step.group?.purpose === "connection" ? executorStep() : null}
            onChange={changeField}
            onBlur={leaveField}
            checkSetup={
              step.key === checkSetupGroup(schema) && previewKind ? (
                <CheckSetupPanel
                  ready={missing.length === 0}
                  pending={discovery.isPending}
                  preview={discovery.data}
                  error={discovery.error && !configIssuesOf(discovery.error) ? discovery.error : null}
                  onCheck={checkSetup}
                />
              ) : null
            }
          />
        ) : (
          <p className="text-sm text-muted-foreground">
            Only the source owner can change its connection and content. You can still change its name, project and schedule.
          </p>
        );
    }
  }

  /**
   * Where it runs, on the connection step of a type that offers the choice,
   * and the device's part: the chooser's devices while a move to My
   * computer waits for one, or the credential step of a Source on a device.
   */
  function executorStep(): ReactNode {
    const onDevice = source !== undefined && storedExecutor === "device" && executor === "device";
    if (executorOptions.length === 0 && !onDevice) return null;
    return (
      <div className="space-y-3">
        {executorOptions.length > 0 ? <WhereItRunsChoice executors={executorOptions} value={executor} onChange={setExecutor} /> : null}
        {onDevice ? <DeviceCredentialStep source={source} /> : executor === "device" ? <DeviceChoicePanel choice={deviceChoice} /> : null}
      </div>
    );
  }

  /**
   * config-check's facts, which state the whole configuration and so show once: on the
   * last content group, the Check setup step, or on the last group when the type has no
   * content group.
   */
  function groupFacts(step: SetupStep): Property[] {
    return step.key === factsGroup ? (configCheck.result?.facts ?? []).map((fact) => ({ label: fact.label, value: fact.value })) : [];
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]">
      <StepRail aria-label="Setup steps" steps={railSteps} current={currentKey} onSelect={openStep} />
      <form
        noValidate
        aria-labelledby="setup-step-heading"
        className="min-w-0 space-y-5"
        onSubmit={(event) => {
          event.preventDefault();
          // A dialog's form, such as New project, submits through the React tree too; only this form saves.
          if (event.target === event.currentTarget) void submit();
        }}
      >
        <div className="space-y-1">
          <h2 id="setup-step-heading" className="text-base font-semibold">
            {current.label}
          </h2>
          {current.group?.description ? <p className="text-sm text-muted-foreground">{current.group.description}</p> : null}
        </div>

        {stepContent(current)}

        {scopeNotice ? (
          <Notice tone={scopeNotice.tone === "danger" ? "danger" : "live"} icon={scopeNotice.tone === "danger" ? AlertTriangle : Info} title={scopeNotice.title}>
            {scopeNotice.message}
          </Notice>
        ) : null}
        {generalIssues.map((message) => (
          <Notice key={message} tone="danger" icon={AlertTriangle} title="This configuration cannot be saved">
            {message}
          </Notice>
        ))}
        {saveError ? (
          <Notice tone="danger" icon={AlertTriangle} title={source ? "Could not save the source" : "Could not create the source"}>
            {scopeChangeErrorMessage(saveError)}
          </Notice>
        ) : null}
        {runningSync.running && !runningSync.confirming ? (
          <Notice tone="warn" icon={RefreshCw} title="A sync of this source is running">
            Saving stops the sync first. What it already saved stays.
          </Notice>
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-4">
          <div className="flex gap-2">
            {previous ? (
              <Button type="button" variant="outline" onClick={() => openStep(previous.key)}>
                Back
              </Button>
            ) : null}
            {next ? (
              <Button type="button" variant="outline" onClick={() => openStep(next.key)}>
                Next
              </Button>
            ) : null}
          </div>
          <div className="flex gap-2">
            <Button type="button" variant="ghost" nativeButton={false} render={<Link to={SOURCES_PATH} />}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving || scopeRefused || awaitsDevice || (source !== undefined && !dirty)}>
              {saving ? "Saving…" : saveLabel}
            </Button>
          </div>
        </div>
      </form>

      <RunningSyncDialog
        open={runningSync.confirming}
        sourceName={source?.name ?? ""}
        stopping={runningSync.stopping}
        error={runningSync.error}
        onStop={runningSync.stopAndSave}
        onWait={runningSync.wait}
      />
      <LeaveDialog blocker={blocker} />
    </div>
  );
}

interface GroupStepProps {
  step: SetupStep;
  schema: ConfigSchema;
  state: ConfigFormState;
  sourceType: string;
  sourceId?: string;
  issues: Readonly<Record<string, string>>;
  notes: Readonly<Record<string, string>>;
  facts: Property[];
  /** Where it runs, on the connection step. */
  executor: ReactNode;
  onChange: (key: string, value: FieldValue) => void;
  onBlur: (key: string) => void;
  checkSetup: ReactNode;
}

/** One of the type's groups: where the Source runs, its fields, what config-check states about the configuration, and "Check setup". */
function GroupStep({ step, schema, state, sourceType, sourceId, issues, notes, facts, executor, onChange, onBlur, checkSetup }: GroupStepProps) {
  return (
    <div className="space-y-5">
      {executor}
      <ConfigGroupFields
        schema={schema}
        group={step.group!}
        state={state}
        sourceType={sourceType}
        sourceId={sourceId}
        issues={issues}
        notes={notes}
        onChange={onChange}
        onBlur={onBlur}
      />
      {facts.length > 0 ? (
        <section aria-label="Detected settings" className="rounded-lg bg-muted/40 p-3">
          <PropertyList items={facts} />
        </section>
      ) : null}
      {checkSetup}
    </div>
  );
}

function RunningSyncDialog({
  open,
  sourceName,
  stopping,
  error,
  onStop,
  onWait,
}: {
  open: boolean;
  sourceName: string;
  stopping: boolean;
  error: unknown;
  onStop: () => void;
  onWait: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={(nextOpen) => (nextOpen ? undefined : onWait())}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Stop the sync and save?</DialogTitle>
          <DialogDescription>
            A sync of {sourceName} is running, and these settings can be saved only once it has ended. Stopping keeps what the
            sync already saved. To keep the sync, wait for it to finish, then save again.
          </DialogDescription>
        </DialogHeader>
        {error ? (
          <p role="alert" className="text-sm text-tone-danger-foreground">
            {errorMessage(error)}
          </p>
        ) : null}
        <DialogFooter>
          <Button variant="outline" onClick={onWait}>
            Wait
          </Button>
          <Button onClick={onStop} disabled={stopping}>
            {stopping ? "Stopping the sync…" : "Stop the sync and save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Asks before an in-app link leaves the page with unsaved changes. */
function LeaveDialog({ blocker }: { blocker: Blocker }) {
  function stay() {
    if (blocker.state === "blocked") blocker.reset();
  }
  function leave() {
    if (blocker.state === "blocked") blocker.proceed();
  }
  return (
    <Dialog open={blocker.state === "blocked"} onOpenChange={(open) => (open ? undefined : stay())}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Leave without saving?</DialogTitle>
          <DialogDescription>Your changes to this source are not saved. They are lost if you leave this page.</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={stay}>
            Keep editing
          </Button>
          <Button variant="destructive" onClick={leave}>
            Leave without saving
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
