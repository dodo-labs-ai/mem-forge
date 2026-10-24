import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { useApi, type components } from "@/api";
import { errorMessage } from "@/lib/errors";
import { newSourcePath, SOURCES_PATH } from "@/lib/paths";
import { BackLink, EmptyState, ErrorNotice, PageHeader } from "@/patterns";
import { Button } from "@/ui/button";
import { Skeleton } from "@/ui/skeleton";
import { configSchemaQuery, useSources, useSourceTypes } from "../api";
import type { Source } from "../model/types";
import { SourceIcon } from "../SourceIcon";
import { SourceSetupForm } from "./SourceSetupForm";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];

function SetupSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading source setup" className="grid gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]">
      <Skeleton className="h-64" />
      <Skeleton className="h-96" />
    </div>
  );
}

/**
 * The form of a type, once its metadata and form description have loaded.
 * A type the service manages is set up by its plugin, not here.
 */
function SetupBody({ gene, source, loadingError, onRetry }: { gene: GeneMetadata | undefined; source?: Source; loadingError: unknown; onRetry: () => void }) {
  const api = useApi();
  const schema = useQuery({ ...configSchemaQuery(api, gene?.name ?? ""), enabled: gene !== undefined && !gene.managed });
  if (loadingError || schema.isError) {
    return <ErrorNotice title="Could not load the source setup" message={errorMessage(loadingError ?? schema.error)} onRetry={onRetry} />;
  }
  if (gene?.managed) {
    return (
      <EmptyState
        title="This source is set up by its plugin"
        description={`${gene.display_name} sources are created and configured by the MemForge plugin of each coding agent.`}
        action={<Button nativeButton={false} render={<Link to={SOURCES_PATH} />}>Back to Sources</Button>}
      />
    );
  }
  if (!gene || !schema.data) return <SetupSkeleton />;
  return <SourceSetupForm key={source?.id ?? gene.name} gene={gene} schema={schema.data} source={source} />;
}

function Heading({ title, type }: { title: string; type?: string }) {
  return (
    <div className="flex items-center gap-3">
      {type ? <SourceIcon type={type} className="size-6" /> : null}
      <PageHeader title={title} />
    </div>
  );
}

/** `/sources/new/:type`: creates a Source of one type. */
export function NewSourcePage() {
  const { type = "" } = useParams<{ type: string }>();
  const types = useSourceTypes();
  const gene = types.data?.find((candidate) => candidate.name === type);

  return (
    <div className="space-y-6">
      <BackLink label="Add source" render={<Link to={newSourcePath()} />} />
      <Heading title={gene ? `Add ${gene.display_name}` : "Add source"} type={gene ? type : undefined} />
      {types.isSuccess && !gene ? (
        <EmptyState
          title="Unknown source type"
          description={`No source type is named “${type}”.`}
          action={<Button nativeButton={false} render={<Link to={newSourcePath()} />}>Choose a source type</Button>}
        />
      ) : (
        <SetupBody gene={gene} loadingError={types.error} onRetry={() => void types.refetch()} />
      )}
    </div>
  );
}

/** `/sources/:sourceId/settings`: edits one Source, opened at the step `?step=` names. */
export function SourceSettingsPage() {
  const { sourceId = "" } = useParams<{ sourceId: string }>();
  const sources = useSources();
  const types = useSourceTypes();
  const source = sources.sources.find((candidate) => candidate.id === sourceId);
  const gene = source ? types.data?.find((candidate) => candidate.name === source.type) : undefined;
  const loaded = sources.isSuccess && types.isSuccess;

  let body;
  if (loaded && !source) {
    body = (
      <EmptyState
        title="Source not found"
        description="It may have been deleted, or it belongs to another workspace."
        action={<Button nativeButton={false} render={<Link to={SOURCES_PATH} />}>Back to Sources</Button>}
      />
    );
  } else if (source && !source.capabilities?.can_configure) {
    body = <EmptyState title="You cannot change this source" description="Ask its owner or a workspace admin to change its settings." />;
  } else {
    body = (
      <SetupBody
        gene={gene}
        source={source}
        loadingError={sources.error ?? types.error}
        onRetry={() => void Promise.all([sources.refetch(), types.refetch()])}
      />
    );
  }

  return (
    <div className="space-y-6">
      <BackLink label="Sources" render={<Link to={SOURCES_PATH} />} />
      <Heading title={source ? `${source.name} settings` : "Source settings"} type={source?.type} />
      {body}
    </div>
  );
}
