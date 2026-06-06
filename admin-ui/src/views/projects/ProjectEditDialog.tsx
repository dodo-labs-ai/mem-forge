/**
 * Dialog for editing an existing project's name and team-wide flag.
 *
 * The project key is immutable once created (it lives in URLs, tags, and
 * memory rows), so the form intentionally exposes only the fields the
 * PATCH /api/projects/{id} endpoint accepts: name and kind.
 */
import { useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import client from "@/api/client";
import type { Project, ProjectKind } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

interface EditFormState {
  name: string;
  shared: boolean;
}

function extractApiErrorDetail(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const candidate = error as { response?: { data?: { detail?: unknown } } };
  const detail = candidate.response?.data?.detail;
  return typeof detail === "string" ? detail : null;
}

export function ProjectEditDialog({
  project,
  open,
  onOpenChange,
}: {
  project: Project;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        {open && (
          <ProjectEditForm
            // Re-keying on the project id guarantees the form state is fresh
            // every time the dialog re-opens for a different project, without
            // a setState-in-effect dance.
            key={project.id}
            project={project}
            onClose={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function ProjectEditForm({
  project,
  onClose,
}: {
  project: Project;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<EditFormState>({
    name: project.name,
    shared: project.kind === "shared",
  });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const updateProject = useMutation({
    mutationFn: async (payload: EditFormState) => {
      const kind: ProjectKind = payload.shared ? "shared" : "normal";
      const body = { name: payload.name.trim(), kind };
      const response = await client.patch<Project>(
        `/api/projects/${project.id}`,
        body,
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      onClose();
    },
    onError: (error: unknown) => {
      const detail = extractApiErrorDetail(error);
      setErrorMessage(detail ?? "Failed to update project.");
    },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (form.name.trim().length === 0) {
      setErrorMessage("Name is required.");
      return;
    }
    updateProject.mutate(form);
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>Edit project</DialogTitle>
        <DialogDescription>
          Rename the project or change whether it's team-wide. The project
          code is fixed once created.
        </DialogDescription>
      </DialogHeader>
      <form onSubmit={submit} className="space-y-4">
        <label className="block space-y-1 text-sm">
          <span className="font-medium">Name</span>
          <Input
            value={form.name}
            onChange={(event) =>
              setForm({ ...form, name: event.target.value })
            }
            autoFocus
          />
        </label>

        <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-sm">
          <span className="block text-xs font-medium text-muted-foreground">
            Code
          </span>
          <span className="font-mono text-sm">{project.key}</span>
          <span className="mt-1 block text-xs text-muted-foreground">
            The code stays the same so existing tags and links keep working.
          </span>
        </div>

        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            className="mt-0.5 size-4 rounded border-border accent-primary"
            checked={form.shared}
            onChange={(event) =>
              setForm({ ...form, shared: event.target.checked })
            }
          />
          <span className="space-y-1">
            <span className="block font-medium">
              Make this a team-wide project
            </span>
            <span className="block text-xs text-muted-foreground">
              Team-wide projects surface for everyone, no matter what
              they're working on.
            </span>
          </span>
        </label>

        {errorMessage && (
          <p className="text-sm text-destructive" role="alert">
            {errorMessage}
          </p>
        )}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={updateProject.isPending}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={updateProject.isPending}>
            {updateProject.isPending && (
              <Loader2 className="size-4 animate-spin" />
            )}
            Save changes
          </Button>
        </DialogFooter>
      </form>
    </>
  );
}
