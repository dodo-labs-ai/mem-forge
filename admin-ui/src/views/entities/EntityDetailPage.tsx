import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { AlertCircle, ArrowLeft, Loader2, Plus, Trash2 } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { EntityDetail as EntityDetailType } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDateTime } from "@/utils/date";

export function EntityDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [newAlias, setNewAlias] = useState("");
  const [mergeTargetId, setMergeTargetId] = useState("");

  const entityQuery = useQuery<EntityDetailType>({
    queryKey: ["entity", id],
    queryFn: () =>
      resourceClient.get(`/entities/${id}`).then((response) => response.data),
    enabled: Boolean(id),
  });

  const addAlias = useMutation({
    mutationFn: (alias: string) =>
      resourceClient.post(`/entities/${id}/aliases`, { alias }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["entity", id] });
      setNewAlias("");
    },
  });

  const deleteAlias = useMutation({
    mutationFn: (alias: string) =>
      resourceClient.delete(
        `/entities/${id}/aliases/${encodeURIComponent(alias)}`,
      ),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["entity", id] }),
  });

  const mergeEntity = useMutation({
    mutationFn: (targetId: string) =>
      resourceClient.post("/entities/merge", {
        source_id: Number(id),
        target_id: Number(targetId),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["entities"] });
      navigate("/entities");
    },
  });

  if (entityQuery.isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading...
      </div>
    );
  }

  if (entityQuery.isError) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border bg-card px-6 py-12 text-center">
        <AlertCircle className="mb-3 size-6 text-destructive" />
        <h1 className="text-sm font-medium">Unable to load entity</h1>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
          {entityQuery.error instanceof Error
            ? entityQuery.error.message
            : "The entity detail request failed."}
        </p>
        <Button
          className="mt-4"
          variant="outline"
          size="sm"
          onClick={() => entityQuery.refetch()}
        >
          Retry
        </Button>
      </div>
    );
  }

  const entity = entityQuery.data;
  if (!entity) {
    return (
      <div className="rounded-xl border bg-card px-6 py-12 text-center text-sm text-muted-foreground">
        Entity not found.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => navigate("/entities")}
        className="-ml-2"
      >
        <ArrowLeft className="mr-1 size-4" /> Back to Entities
      </Button>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center gap-3">
            <CardTitle className="text-xl">{entity.canonical_name}</CardTitle>
          </div>
          {entity.display_name &&
            entity.display_name !== entity.canonical_name && (
              <p className="text-sm text-muted-foreground">
                {entity.display_name}
              </p>
            )}
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            {entity.linked_memory_count} linked memories
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            Aliases ({entity.aliases.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              const next = newAlias.trim();
              if (next) addAlias.mutate(next);
            }}
            className="mb-4 flex gap-2"
          >
            <Input
              value={newAlias}
              onChange={(event) => setNewAlias(event.target.value)}
              placeholder="Add new alias..."
              className="flex-1"
            />
            <Button
              type="submit"
              size="default"
              disabled={!newAlias.trim() || addAlias.isPending}
            >
              <Plus className="mr-1 size-4" /> Add
            </Button>
          </form>

          {entity.aliases.length === 0 ? (
            <p className="text-sm text-muted-foreground">No aliases yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Alias</TableHead>
                  <TableHead className="w-32">Source</TableHead>
                  <TableHead className="w-32">Added</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {entity.aliases.map((alias) => (
                  <TableRow key={alias.alias}>
                    <TableCell className="text-foreground">
                      {alias.alias}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {alias.source}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {alias.created_at
                        ? formatDateTime(alias.created_at)
                        : "-"}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => deleteAlias.mutate(alias.alias)}
                        title="Delete alias"
                      >
                        <Trash2 className="size-3.5 text-muted-foreground hover:text-destructive" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Merge into another entity</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-3 text-xs text-muted-foreground">
            This will make the current entity an alias of the target.
          </p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
            }}
            className="flex gap-2"
          >
            <Input
              value={mergeTargetId}
              onChange={(event) => setMergeTargetId(event.target.value)}
              className="w-40"
              placeholder="Target entity ID"
            />
            <Dialog>
              <DialogTrigger
                render={
                  <Button
                    variant="destructive"
                    disabled={!mergeTargetId.trim() || mergeEntity.isPending}
                  />
                }
              >
                Merge
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Confirm merge</DialogTitle>
                  <DialogDescription>
                    This will merge "{entity.canonical_name}" into entity #
                    {mergeTargetId}. This action cannot be undone.
                  </DialogDescription>
                </DialogHeader>
                <DialogFooter>
                  <Button
                    variant="destructive"
                    onClick={() => mergeEntity.mutate(mergeTargetId.trim())}
                    disabled={mergeEntity.isPending}
                  >
                    {mergeEntity.isPending ? "Merging..." : "Confirm merge"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
