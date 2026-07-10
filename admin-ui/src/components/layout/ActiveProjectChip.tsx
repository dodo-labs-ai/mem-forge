import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";

import { resourceClient } from "@/api/client";
import { isReservedProjectKey } from "@/api/projectKeys";
import type { Project } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { useActiveProject } from "@/state/activeProject";

/**
 * Reserved buckets are system mechanics, not user-facing projects, so the
 * picker never lists them. Project rows in the picker are real projects only.
 */
const SEARCH_THRESHOLD = 5;

const CHIP_LABEL_PICK = "Pick a project";
const CHIP_LABEL_CROSS = "Cross-project view";
const CHIP_LABEL_PREFIX = "Working in:";
const CROSS_PROJECT_LABEL = "Cross-project view";
const CROSS_PROJECT_SUFFIX = "(admin)";

function ChipLabel({
  activeProjectName,
  crossProjectMode,
}: {
  activeProjectName: string | null;
  crossProjectMode: boolean;
}) {
  if (crossProjectMode) {
    return (
      <span className="italic text-muted-foreground">{CHIP_LABEL_CROSS}</span>
    );
  }
  if (!activeProjectName) {
    return (
      <span className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="size-1.5 rounded-full bg-amber-500"
        />
        <span className="text-muted-foreground">{CHIP_LABEL_PICK}</span>
      </span>
    );
  }
  return (
    <span className="flex items-baseline gap-1">
      <span className="text-muted-foreground">{CHIP_LABEL_PREFIX}</span>
      <span className="font-medium text-foreground">{activeProjectName}</span>
    </span>
  );
}

export function ActiveProjectChip() {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const {
    activeProjectKey,
    crossProjectMode,
    setActiveProjectKey,
    enableCrossProjectMode,
  } = useActiveProject();

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () => resourceClient.get<Project[]>("/projects").then((r) => r.data),
  });

  const userProjects = useMemo(() => {
    const projects = projectsQuery.data ?? [];
    return projects.filter((p) => !isReservedProjectKey(p.key));
  }, [projectsQuery.data]);

  const activeProject = useMemo(
    () => userProjects.find((p) => p.key === activeProjectKey) ?? null,
    [userProjects, activeProjectKey],
  );

  const filteredProjects = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return userProjects;
    return userProjects.filter(
      (p) =>
        p.name.toLowerCase().includes(needle) ||
        p.key.toLowerCase().includes(needle),
    );
  }, [userProjects, filter]);

  const showSearch = userProjects.length > SEARCH_THRESHOLD;

  const handlePick = (key: string) => {
    setActiveProjectKey(key);
    setFilter("");
    setOpen(false);
  };

  const handleCrossProject = () => {
    enableCrossProjectMode();
    setFilter("");
    setOpen(false);
  };

  return (
    <PopoverPrimitive.Root open={open} onOpenChange={setOpen}>
      <PopoverPrimitive.Trigger
        render={
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="gap-1.5"
            aria-label="Active project"
          />
        }
      >
        <ChipLabel
          activeProjectName={activeProject?.name ?? null}
          crossProjectMode={crossProjectMode}
        />
        <ChevronDown className="size-3.5 opacity-70" />
      </PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Positioner sideOffset={6} align="end">
          <PopoverPrimitive.Popup
            className={cn(
              "z-50 w-72 rounded-lg border bg-popover p-1.5 text-sm text-popover-foreground shadow-md outline-none",
              "data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",
            )}
          >
            {showSearch && (
              <div className="p-1.5">
                <Input
                  type="search"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="Filter projects"
                  aria-label="Filter projects"
                />
              </div>
            )}
            <div className="max-h-72 overflow-y-auto">
              {projectsQuery.isLoading && (
                <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                  Loading projects…
                </div>
              )}
              {!projectsQuery.isLoading && filteredProjects.length === 0 && (
                <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                  {userProjects.length === 0
                    ? "No projects yet."
                    : "No matches."}
                </div>
              )}
              {filteredProjects.map((project) => {
                const isActive =
                  !crossProjectMode && project.key === activeProjectKey;
                return (
                  <button
                    key={project.id}
                    type="button"
                    onClick={() => handlePick(project.key)}
                    className={cn(
                      "flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-muted",
                      isActive && "bg-muted",
                    )}
                  >
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {project.name}
                    </span>
                    <span className="shrink-0 text-[11px] text-muted-foreground/80">
                      {project.key}
                    </span>
                  </button>
                );
              })}
            </div>
            <Separator className="my-1" />
            <button
              type="button"
              onClick={handleCrossProject}
              className={cn(
                "flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left italic text-muted-foreground transition-colors hover:bg-muted",
                crossProjectMode && "bg-muted",
              )}
            >
              <span>{CROSS_PROJECT_LABEL}</span>
              <span className="text-xs not-italic text-muted-foreground/70">
                {CROSS_PROJECT_SUFFIX}
              </span>
            </button>
          </PopoverPrimitive.Popup>
        </PopoverPrimitive.Positioner>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
