import { ExternalLink } from "lucide-react";
import type { Project } from "@/api/types";

/** Where the image serves the new admin UI, beside this one at `/`. */
const NEW_ADMIN_BASE_PATH = "/v2/";

/**
 * @deprecated admin-ui-v1: projects are deleted in the new admin UI, which
 * states which memories move and which sources stop writing to the project,
 * then asks for the project code (ADR 0044). This link opens the project
 * there instead of deleting it here.
 */
export function DeleteInNewAdminLink({ project }: { project: Pick<Project, "key" | "name"> }) {
  return (
    <a
      href={`${NEW_ADMIN_BASE_PATH}projects/${encodeURIComponent(project.key)}`}
      aria-label={`Delete ${project.name} in the new admin`}
      className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
    >
      Delete in new admin
      <ExternalLink className="size-3" />
    </a>
  );
}
