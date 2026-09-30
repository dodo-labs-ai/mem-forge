import { AlertCircle, ArrowLeft, Lock, RefreshCw, SearchX } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { HTTP_STATUS, isApiErrorStatus, useProjects, useWorkspaceTarget } from "@/api";
import { useSourceTypeLabels } from "@/features/sources";
import { errorMessage } from "@/lib/errors";
import { formatDate, formatDateTime } from "@/lib/format";
import { MEMORIES_PATH } from "@/lib/paths";
import { BackLink, EmptyState, PropertyList, type Property } from "@/patterns";
import { Button, buttonVariants } from "@/ui/button";
import { Skeleton } from "@/ui/skeleton";
import { useMemory, useOpenReviews } from "./api";
import { DetailCard } from "./DetailCard";
import { EvidenceSection } from "./EvidenceSection";
import { LifecycleCard } from "./LifecycleCard";
import { MemoryActions } from "./MemoryActions";
import { MemoryStatusLabel, MemoryTypeBadge, ProjectLabel } from "./MemoryBadges";
import { MemoryStatusBanner } from "./MemoryStatusBanner";
import { lifecycleDetail, statusBanner } from "./model/lifecycle";
import { PRIVATE_VISIBILITY, sourceName } from "./model/memoryRows";
import { projectBadge } from "./model/projects";
import type { MemoryDetail, Project } from "./model/types";
import { RelatedMemoriesCard } from "./RelatedMemoriesCard";

const EMPTY_PROJECTS: Project[] = [];

function BackToMemories() {
  return <BackLink label="Memories" render={<Link to={MEMORIES_PATH} />} />;
}

function AccessValue({ memory }: { memory: MemoryDetail }) {
  const workspace = useWorkspaceTarget()?.label;
  if (memory.visibility === PRIVATE_VISIBILITY) {
    return (
      <span className="inline-flex items-center gap-1.5">
        <Lock aria-hidden className="size-3.5 text-muted-foreground" />
        Only me
      </span>
    );
  }
  return <>{workspace ? `Everyone in ${workspace}` : "Everyone in this workspace"}</>;
}

interface MemoryDetailsCardProps {
  memory: MemoryDetail;
  projects: readonly Project[];
  typeLabels: Readonly<Record<string, string>>;
}

function MemoryDetailsCard({ memory, projects, typeLabels }: MemoryDetailsCardProps) {
  const project = projectBadge(memory.project_key, projects);
  const sources = (memory.sources ?? []).map((source) => sourceName(source, typeLabels));
  const items: Property[] = [
    { label: "Type", value: <MemoryTypeBadge type={memory.memory_type} /> },
    { label: "Status", value: <MemoryStatusLabel status={memory.status} /> },
    { label: "Access", value: <AccessValue memory={memory} /> },
    { label: "Project", value: project ? <ProjectLabel project={project} /> : "None" },
    { label: "Sources", value: sources.length > 0 ? sources.join(", ") : "None" },
    ...(memory.created_at ? [{ label: "Created", value: formatDate(memory.created_at) }] : []),
    ...(memory.updated_at ? [{ label: "Updated", value: formatDateTime(memory.updated_at) }] : []),
  ];
  return (
    <DetailCard title="Details">
      <PropertyList items={items} />
    </DetailCard>
  );
}

function EntitiesCard({ entities }: { entities: readonly string[] }) {
  if (entities.length === 0) return null;
  return (
    <DetailCard title="Entities">
      <ul className="flex flex-wrap gap-1.5">
        {entities.map((entity) => (
          <li key={entity} className="rounded-md bg-muted px-2 py-0.5 text-xs text-subtle-foreground">
            {entity}
          </li>
        ))}
      </ul>
    </DetailCard>
  );
}

function DetailSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading memory" className="space-y-4">
      <Skeleton className="h-5 w-40" />
      <Skeleton className="h-8 w-3/4" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

export function MemoryDetailPage() {
  const { memoryId = "" } = useParams<{ memoryId: string }>();
  const memoryQuery = useMemory(memoryId);
  const reviews = useOpenReviews();
  const projects = useProjects();
  const typeLabels = useSourceTypeLabels();

  if (memoryQuery.isPending) {
    return (
      <div className="space-y-4">
        <BackToMemories />
        <DetailSkeleton />
      </div>
    );
  }

  if (memoryQuery.isError) {
    const notFound = isApiErrorStatus(memoryQuery.error, HTTP_STATUS.notFound);
    return (
      <div className="space-y-4">
        <BackToMemories />
        <div className="rounded-lg border bg-surface">
          {notFound ? (
            <EmptyState
              icon={SearchX}
              title="Memory not found"
              description="It may have been removed, or it belongs to a source you cannot access. Private memories are visible only to their owner."
              action={
                <Link to={MEMORIES_PATH} className={buttonVariants({ variant: "outline" })}>
                  <ArrowLeft />
                  Back to Memories
                </Link>
              }
            />
          ) : (
            <EmptyState
              icon={AlertCircle}
              title="Unable to load memory"
              description={errorMessage(memoryQuery.error)}
              action={
                <Button variant="outline" onClick={() => void memoryQuery.refetch()}>
                  <RefreshCw />
                  Retry
                </Button>
              }
            />
          )}
        </div>
      </div>
    );
  }

  const memory = memoryQuery.data;
  const banner = statusBanner(memory);
  const lifecycle = lifecycleDetail(memory);

  return (
    <div className="space-y-4">
      <BackToMemories />
      {banner ? <MemoryStatusBanner banner={banner} /> : null}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-4">
          <div className="flex flex-wrap items-center gap-2.5">
            <MemoryTypeBadge type={memory.memory_type} />
            <MemoryStatusLabel status={memory.status} />
            <div className="ml-auto">
              <MemoryActions memory={memory} reviewId={reviews.reviewByMemory.get(memory.id)} />
            </div>
          </div>
          <h1 className="text-xl leading-snug font-semibold tracking-tight text-foreground">{memory.content}</h1>
          {lifecycle ? <LifecycleCard detail={lifecycle} /> : null}
          <RelatedMemoriesCard memory={memory} typeLabels={typeLabels} />
          <EvidenceSection evidence={memory.evidence ?? []} typeLabels={typeLabels} />
        </div>
        <aside className="space-y-4">
          <MemoryDetailsCard memory={memory} projects={projects.data ?? EMPTY_PROJECTS} typeLabels={typeLabels} />
          <EntitiesCard entities={memory.entity_refs ?? []} />
        </aside>
      </div>
    </div>
  );
}
