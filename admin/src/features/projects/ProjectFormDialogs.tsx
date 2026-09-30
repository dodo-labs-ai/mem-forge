import { zodResolver } from "@hookform/resolvers/zod";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useId, useState, type ReactNode } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { errorMessage } from "@/lib/errors";
import { ErrorNotice } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Input } from "@/ui/input";
import { Label } from "@/ui/label";
import { useCreateProject, useRenameProject } from "./api";
import {
  createProjectBody,
  createProjectSchema,
  editProjectSchema,
  type CreateProjectInput,
  type CreateProjectValues,
  type EditProjectInput,
  type EditProjectValues,
} from "./model/projectForm";
import type { Project } from "./model/types";

const DIALOG_WIDTH = "sm:max-w-lg";

function Field({
  id,
  label,
  required = false,
  hint,
  error,
  children,
}: {
  id: string;
  label: string;
  required?: boolean;
  hint?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>
        {label}
        {required ? <span aria-hidden className="text-tone-danger">*</span> : null}
      </Label>
      {children}
      {error ? (
        <p id={`${id}-error`} role="alert" className="text-xs text-tone-danger-foreground">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

export function CreateProjectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={DIALOG_WIDTH}>
        {open ? <CreateProjectForm onClose={() => onOpenChange(false)} /> : null}
      </DialogContent>
    </Dialog>
  );
}

function CreateProjectForm({ onClose }: { onClose: () => void }) {
  const ids = useId();
  const nameId = `${ids}-name`;
  const codeId = `${ids}-code`;
  const createProject = useCreateProject();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const form = useForm<CreateProjectInput, unknown, CreateProjectValues>({
    resolver: zodResolver(createProjectSchema),
    defaultValues: { name: "", code: "" },
    mode: "onChange",
  });
  const { errors, isSubmitting } = form.formState;
  // A refused code must stay visible, so the section cannot close over it.
  const showAdvanced = advancedOpen || errors.code !== undefined;

  const submit = form.handleSubmit(async (values) => {
    setServerError(null);
    try {
      const created = await createProject.mutateAsync(createProjectBody(values));
      toast.success(`Project “${created.name}” created`);
      onClose();
    } catch (error) {
      setServerError(errorMessage(error));
    }
  });

  return (
    <form onSubmit={submit} noValidate className="grid gap-4">
      <DialogHeader>
        <DialogTitle>New project</DialogTitle>
        <DialogDescription>
          Name it after what the memories are about, for example a product, codebase, or initiative.
        </DialogDescription>
      </DialogHeader>
      <Field id={nameId} label="Name" required error={errors.name?.message}>
        <Input
          id={nameId}
          autoFocus
          aria-required
          aria-invalid={errors.name ? true : undefined}
          aria-describedby={errors.name ? `${nameId}-error` : undefined}
          {...form.register("name")}
        />
      </Field>
      <div className="border-t pt-3">
        <button
          type="button"
          aria-expanded={showAdvanced}
          aria-controls={`${ids}-advanced`}
          onClick={() => setAdvancedOpen((current) => !current)}
          className="flex items-center gap-1.5 text-sm font-medium text-subtle-foreground hover:text-foreground"
        >
          {showAdvanced ? <ChevronDown aria-hidden className="size-4" /> : <ChevronRight aria-hidden className="size-4" />}
          Advanced
        </button>
        {showAdvanced ? (
          <div id={`${ids}-advanced`} className="pt-3">
            <Field
              id={codeId}
              label="Code (optional)"
              hint="Short code shown in URLs and project labels. Created from the name when left empty. It cannot change later."
              error={errors.code?.message}
            >
              <Input
                id={codeId}
                placeholder="PAY"
                className="font-mono"
                aria-invalid={errors.code ? true : undefined}
                aria-describedby={errors.code ? `${codeId}-error` : `${codeId}-hint`}
                {...form.register("code")}
              />
            </Field>
          </div>
        ) : null}
      </div>
      {serverError ? <ErrorNotice title="Could not create the project" message={serverError} /> : null}
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSubmitting || errors.code !== undefined}>
          Create
        </Button>
      </DialogFooter>
    </form>
  );
}

export function EditProjectDialog({
  project,
  onOpenChange,
}: {
  /** The project to rename; the dialog is open while this is set. */
  project: Project | null;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={project !== null} onOpenChange={onOpenChange}>
      <DialogContent className={DIALOG_WIDTH}>
        {project ? <EditProjectForm key={project.id} project={project} onClose={() => onOpenChange(false)} /> : null}
      </DialogContent>
    </Dialog>
  );
}

function EditProjectForm({ project, onClose }: { project: Project; onClose: () => void }) {
  const ids = useId();
  const nameId = `${ids}-name`;
  const renameProject = useRenameProject();
  const [serverError, setServerError] = useState<string | null>(null);
  const form = useForm<EditProjectInput, unknown, EditProjectValues>({
    resolver: zodResolver(editProjectSchema),
    defaultValues: { name: project.name },
  });
  const { errors, isSubmitting } = form.formState;

  const submit = form.handleSubmit(async ({ name }) => {
    setServerError(null);
    try {
      await renameProject.mutateAsync({ projectId: project.id, name });
      toast.success(`Project renamed to “${name}”`);
      onClose();
    } catch (error) {
      setServerError(errorMessage(error));
    }
  });

  return (
    <form onSubmit={submit} noValidate className="grid gap-4">
      <DialogHeader>
        <DialogTitle>Edit project</DialogTitle>
        <DialogDescription>Rename the project. The code stays the same so existing links keep working.</DialogDescription>
      </DialogHeader>
      <Field id={nameId} label="Name" required error={errors.name?.message}>
        <Input
          id={nameId}
          autoFocus
          aria-required
          aria-invalid={errors.name ? true : undefined}
          aria-describedby={errors.name ? `${nameId}-error` : undefined}
          {...form.register("name")}
        />
      </Field>
      <div className="space-y-1.5">
        <p className="text-sm font-medium">Code</p>
        <p className="rounded-lg border bg-surface-subtle px-2.5 py-1.5 font-mono text-sm text-subtle-foreground">
          {project.key}
        </p>
      </div>
      {serverError ? <ErrorNotice title="Could not rename the project" message={serverError} /> : null}
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSubmitting}>
          Save changes
        </Button>
      </DialogFooter>
    </form>
  );
}
