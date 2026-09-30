import { pluralize } from "@/lib/format";

/**
 * The sentence about the Sources the delete stops writing to the project,
 * before or after it; empty when there are none. The server removes a fixed
 * binding to the project and drops a field binding's mappings to it, so both
 * kinds are described by what they stop doing, not as unbound.
 */
function releasedSourcesSentence(count: number, phase: "before" | "after"): string {
  if (count === 0) return "";
  return ` ${pluralize(count, "source")} ${phase === "before" ? "will stop" : "stopped"} writing to this project.`;
}

/** What deleting a project does, stated before the user confirms. */
export function deleteProjectDescription(memoryCount: number, sourceCount: number): string {
  const memories =
    memoryCount === 0
      ? "The project has no memories."
      : `${pluralize(memoryCount, "memory", "memories")} ${memoryCount === 1 ? "moves" : "move"} to the Unsorted project.`;
  return `${memories}${releasedSourcesSentence(sourceCount, "before")} This cannot be undone.`;
}

/** The notice after a delete, with what the server moved and released. */
export function deletedProjectNotice(name: string, movedCount: number, releasedCount: number): string {
  const moved = `${pluralize(movedCount, "memory", "memories")} moved to the Unsorted project.`;
  return `Project “${name}” deleted. ${moved}${releasedSourcesSentence(releasedCount, "after")}`;
}

/** Navigation state that carries the notice to the list after a delete on the detail page. */
export interface DeletedProjectState {
  deletedNotice: string;
}

export function deletedNoticeFrom(state: unknown): string | null {
  if (state && typeof state === "object" && "deletedNotice" in state) {
    const notice = (state as DeletedProjectState).deletedNotice;
    if (typeof notice === "string") return notice;
  }
  return null;
}
