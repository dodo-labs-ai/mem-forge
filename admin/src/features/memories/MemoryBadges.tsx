import { FileText, FolderKanban, GitBranch, ListChecks, Lock, Ruler, type LucideIcon } from "lucide-react";
import { SourceIcon } from "@/features/sources";
import { cn } from "@/lib/cn";
import { StatusBadge } from "@/patterns";
import {
  memoryStatusLabel,
  memoryStatusTone,
  memoryTypeLabel,
  memoryTypeTone,
  type MemoryType,
} from "./model/memoryPresentation";
import type { ProjectBadge } from "./model/projects";
import { RELATION_LABEL_NAMES, RELATION_LABEL_TONES } from "./model/relations";
import type { MemorySourceRef, RelationLabel } from "./model/types";

const MEMORY_TYPE_ICONS: Record<MemoryType, LucideIcon> = {
  fact: FileText,
  decision: GitBranch,
  convention: Ruler,
  procedure: ListChecks,
};

export function MemoryTypeBadge({ type }: { type: string }) {
  const Icon = MEMORY_TYPE_ICONS[type as MemoryType] ?? FileText;
  return (
    <StatusBadge tone={memoryTypeTone(type)}>
      <Icon aria-hidden />
      {memoryTypeLabel(type)}
    </StatusBadge>
  );
}

export function MemoryStatusLabel({ status }: { status: string }) {
  return (
    <StatusBadge tone={memoryStatusTone(status)} variant="dot">
      {memoryStatusLabel(status)}
    </StatusBadge>
  );
}

export function RelationLabelBadge({ label }: { label: RelationLabel }) {
  return <StatusBadge tone={RELATION_LABEL_TONES[label]}>{RELATION_LABEL_NAMES[label]}</StatusBadge>;
}

const UNSORTED_TITLE = "Unsorted project (built-in catch-all)";

export function ProjectLabel({ project }: { project: ProjectBadge }) {
  if (project.kind === "shared") return <StatusBadge tone="live">{project.label}</StatusBadge>;
  return (
    <span
      className="inline-flex items-center gap-1 text-xs text-subtle-foreground"
      title={project.kind === "unsorted" ? UNSORTED_TITLE : undefined}
    >
      <FolderKanban aria-hidden className="size-3 text-muted-foreground" />
      {project.label}
    </span>
  );
}

export function OnlyMeLabel() {
  return (
    <span className="inline-flex items-center gap-1 text-xs text-subtle-foreground">
      <Lock aria-hidden className="size-3 text-muted-foreground" />
      Only me
    </span>
  );
}

interface SourceLabelProps {
  source: MemorySourceRef;
  name: string;
  className?: string;
}

export function SourceLabel({ source, name, className }: SourceLabelProps) {
  return (
    <span className={cn("inline-flex min-w-0 items-center gap-1.5 text-xs text-subtle-foreground", className)}>
      <SourceIcon type={source.source_type} className="size-3.5" />
      <span className="truncate">{name}</span>
    </span>
  );
}
