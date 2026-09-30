import { ArrowRight, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { projectMemoriesPath } from "@/lib/paths";
import { Button } from "@/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/ui/dropdown-menu";
import type { Project } from "./model/types";

interface ProjectRowActionsProps {
  project: Project;
  /** The edit and delete actions; `null` when the caller may not change projects. */
  changes: { onEdit: (project: Project) => void; onDelete: (project: Project) => void } | null;
}

export function ProjectRowActions({ project, changes }: ProjectRowActionsProps) {
  const navigate = useNavigate();
  return (
    <div className="flex justify-end" onClick={(event) => event.stopPropagation()}>
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`Actions for ${project.name}`} />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuItem onClick={() => navigate(projectMemoriesPath(project.key))}>
            <ArrowRight />
            View memories
          </DropdownMenuItem>
          {changes ? (
            <>
              <DropdownMenuItem onClick={() => changes.onEdit(project)}>
                <Pencil />
                Edit
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onClick={() => changes.onDelete(project)}>
                <Trash2 />
                Delete…
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
