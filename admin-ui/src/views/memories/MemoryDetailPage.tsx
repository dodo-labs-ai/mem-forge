import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, ArrowLeft, ExternalLink, Loader2 } from "lucide-react";
import client from "@/api/client";
import type { Memory, MemorySource } from "@/api/types";
import { ConfidenceBadge, MemoryTypeBadge, StatusDot } from "@/components/admin/StatusBadge";
import { MemoryTypeIcon } from "@/components/memories/MemoryTypeIcon";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { formatDateTime, timeAgo } from "@/utils/date";
import { getLifecycleDetail, type LifecycleDetail } from "./lifecycle";

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  superseded: "Superseded",
  retired: "Retired",
  decayed: "Retired",
  pending_review: "Needs Review",
};

export function MemoryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const memoryQuery = useQuery<Memory>({
    queryKey: ["memory", id],
    queryFn: () =>
      client
        .get(`/api/memories/${id}`, { params: { include_private: "true" } })
        .then((response) => response.data),
    enabled: Boolean(id),
  });

  const updateStatus = useMutation({
    mutationFn: (status: string) => client.put(`/api/memories/${id}`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memory", id] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  if (memoryQuery.isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading...
      </div>
    );
  }

  if (memoryQuery.isError) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border bg-card px-6 py-12 text-center">
        <AlertCircle className="mb-3 size-6 text-destructive" />
        <h1 className="text-sm font-medium">Unable to load memory</h1>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
          {memoryQuery.error instanceof Error
            ? memoryQuery.error.message
            : "The memory detail request failed."}
        </p>
        <Button
          className="mt-4"
          variant="outline"
          size="sm"
          onClick={() => memoryQuery.refetch()}
        >
          Retry
        </Button>
      </div>
    );
  }

  const memory = memoryQuery.data;
  if (!memory) {
    return (
      <div className="rounded-xl border bg-card px-6 py-12 text-center text-sm text-muted-foreground">
        Memory not found.
      </div>
    );
  }

  const lifecycleDetail = getLifecycleDetail(memory);
  const origin = memory.origin_source_type;

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/memories")} className="-ml-2">
        <ArrowLeft className="mr-1 size-4" /> Back to Memories
      </Button>

      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3">
              <StatusDot status={memory.status} />
              {origin ? (
                <SourceIcon type={origin} client={memory.origin_client} className="size-4" />
              ) : (
                <MemoryTypeIcon type={memory.memory_type} className="size-4" />
              )}
              <MemoryTypeBadge type={memory.memory_type} />
              <span className="text-xs text-muted-foreground">
                {STATUS_LABELS[memory.status] ?? memory.status}
              </span>
            </div>
            {memory.status === "active" && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={updateStatus.isPending}
                  onClick={() => updateStatus.mutate("retired")}
                >
                  Retire
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <p className="mb-6 text-base leading-relaxed text-foreground">{memory.content}</p>

          <div className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
            <div>
              <span className="text-xs text-muted-foreground">Confidence</span>
              <div className="mt-1">
                <ConfidenceBadge confidence={memory.confidence} />
              </div>
            </div>
            <div>
              <span className="text-xs text-muted-foreground">Corroborations</span>
              <div className="mt-1 text-foreground">{memory.corroboration_count}</div>
            </div>
            <div>
              <span className="text-xs text-muted-foreground">Created</span>
              <div className="mt-1 text-foreground">{timeAgo(memory.created_at)}</div>
            </div>
            <div>
              <span className="text-xs text-muted-foreground">Updated</span>
              <div className="mt-1 text-foreground">{formatDateTime(memory.updated_at)}</div>
            </div>
          </div>

          {memory.tags.length > 0 && (
            <>
              <Separator className="my-4" />
              <div className="flex flex-wrap gap-1.5">
                {memory.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </>
          )}

          {memory.entity_refs.length > 0 && (
            <>
              <Separator className="my-4" />
              <span className="mb-2 block text-xs text-muted-foreground">Entities</span>
              <div className="flex flex-wrap gap-2">
                {memory.entity_refs.map((ref) => (
                  <Link
                    key={ref}
                    to={`/entities?search=${encodeURIComponent(ref)}`}
                    className="text-sm text-primary hover:underline"
                  >
                    {ref}
                  </Link>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {lifecycleDetail && <LifecycleCard detail={lifecycleDetail} />}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Provenance ({memory.sources.length} sources)</CardTitle>
        </CardHeader>
        <CardContent>
          {memory.sources.length === 0 ? (
            <p className="text-sm text-muted-foreground">No provenance sources recorded.</p>
          ) : (
            <ProvenanceSections sources={memory.sources} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LifecycleCard({ detail }: { detail: LifecycleDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Lifecycle</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-3 text-sm sm:grid-cols-[10rem_1fr]">
          <dt className="text-xs text-muted-foreground">Status</dt>
          <dd className="text-foreground">{detail.status}</dd>

          <dt className="text-xs text-muted-foreground">Reason</dt>
          <dd className="text-foreground">{detail.reason}</dd>

          {detail.occurredLabel && detail.occurredAt && (
            <>
              <dt className="text-xs text-muted-foreground">{detail.occurredLabel}</dt>
              <dd className="text-foreground">{formatDateTime(detail.occurredAt)}</dd>
            </>
          )}

          {detail.replacedBy && (
            <>
              <dt className="text-xs text-muted-foreground">Replaced by</dt>
              <dd>
                <Link to={`/memories/${detail.replacedBy}`} className="text-primary hover:underline">
                  {detail.replacedBy}
                </Link>
              </dd>
            </>
          )}

          {detail.technicalReason && (
            <>
              <dt className="text-xs text-muted-foreground">Technical reason</dt>
              <dd className="text-xs text-muted-foreground">{detail.technicalReason}</dd>
            </>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function ProvenanceSections({ sources }: { sources: MemorySource[] }) {
  const extracted = sources.filter((src) => src.support_kind !== "corroborated");
  const corroborated = sources.filter((src) => src.support_kind === "corroborated");

  return (
    <div className="space-y-6">
      {extracted.length > 0 && (
        <section className="space-y-3">
          <h3 className="text-xs font-medium uppercase text-muted-foreground">
            Extracted from
          </h3>
          <div className="space-y-4">
            {extracted.map((src, index) => (
              <SourceCard key={`extracted-${src.doc_id}-${index}`} source={src} />
            ))}
          </div>
        </section>
      )}
      {corroborated.length > 0 && (
        <section className="space-y-3">
          <h3 className="text-xs font-medium uppercase text-muted-foreground">
            Also supported by
          </h3>
          <div className="space-y-4">
            {corroborated.map((src, index) => (
              <SourceCard key={`corroborated-${src.doc_id}-${index}`} source={src} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function SourceCard({ source }: { source: MemorySource }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="mb-2 flex items-center gap-2">
        <Badge variant="secondary">{source.source_type}</Badge>
        <span className="text-sm font-medium text-foreground">
          {source.doc_title ?? source.doc_id}
        </span>
        {source.source_url && (
          <a
            href={source.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="size-3.5" />
          </a>
        )}
      </div>
      {source.excerpt && (
        <p className="mb-2 border-l-2 border-border pl-3 text-sm italic text-muted-foreground">
          "{source.excerpt}"
        </p>
      )}
      <div className="text-xs text-muted-foreground">
        Observed: {formatDateTime(source.added_at)}
      </div>
    </div>
  );
}
