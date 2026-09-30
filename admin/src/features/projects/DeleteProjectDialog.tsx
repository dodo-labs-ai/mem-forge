import { errorMessage } from "@/lib/errors";
import { ConfirmDialog } from "@/patterns";
import { useDeleteProject, useProjectDeletionImpact } from "./api";
import { deleteProjectDescription, deletedProjectNotice } from "./model/projectDeletion";
import type { Project } from "./model/types";

interface DeleteProjectDialogProps {
  /** The project to delete; the dialog is open while this is set. */
  project: Project | null;
  onOpenChange: (open: boolean) => void;
  /** Receives the notice that states how many memories moved and how many sources stopped writing to the project. */
  onDeleted: (notice: string) => void;
}

/**
 * The one delete confirmation for projects, used by the list and the detail
 * page. It states which memories move and which sources stop writing to the
 * project, then asks for the project code before deleting.
 */
export function DeleteProjectDialog({ project, onOpenChange, onDeleted }: DeleteProjectDialogProps) {
  const deleteProject = useDeleteProject();
  const impact = useProjectDeletionImpact(project?.id ?? null);
  const description = impact.isSuccess
    ? deleteProjectDescription(impact.data.memory_count, impact.data.source_count)
    : impact.isError
      ? `Could not check what the delete changes. ${errorMessage(impact.error)}`
      : "Checking what the delete changes…";
  return (
    <ConfirmDialog
      open={project !== null}
      onOpenChange={onOpenChange}
      title={`Delete ${project?.name ?? "project"}?`}
      description={description}
      continueLabel="Continue"
      confirmationText={project?.key}
      confirmLabel="Delete project"
      tone="danger"
      ready={impact.isSuccess && !impact.isFetching}
      onConfirm={async () => {
        if (!project) return;
        const deletion = await deleteProject.mutateAsync(project.id);
        onDeleted(deletedProjectNotice(project.name, deletion.rebucketed_count, deletion.released_source_count));
      }}
    />
  );
}
