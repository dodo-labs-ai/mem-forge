import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { projectQueryKeys, unwrap, useApi, type components } from "@/api";
import { localSyncKeys } from "@/features/local-sync";
import { sourceKeys } from "../api";
import { useLocalSetupJob } from "../useLocalSetupJob";
import type { CheckSetupKind } from "./model/checkSetup";

type Schemas = components["schemas"];
export type ConfigCheck = Schemas["ConfigCheckResponse"];
export type CreateSourceBody = Schemas["CreateSourceRequest"];
export type UpdateSourceBody = Schemas["UpdateSourceRequest"];
export type ScopeChangePreview = Schemas["SourceScopeChangePreviewResponse"];

/** How many items "Check setup" lists. */
export const CHECK_SETUP_LIMIT = 5;
/** How long the settings page waits after the last edit before it asks what saving would do to the scope. */
const SCOPE_PREVIEW_DELAY_MS = 400;

/**
 * Runs config-check, the server's save steps on a copy of the form's
 * configuration, which return the values save would derive, facts and field
 * issues. `result` is the answer to the latest check, so an earlier answer
 * that arrives late never replaces a later one. `check` also returns its
 * answer, for the check before a save. With `sourceId` the server reads the
 * stored configuration for kept secrets and fixed fields.
 */
export function useConfigCheck(sourceType: string, sourceId: string | undefined) {
  const api = useApi();
  const latest = useRef(0);
  const [result, setResult] = useState<ConfigCheck | null>(null);
  const check = useCallback(
    async (config: Record<string, unknown>) => {
      const ticket = ++latest.current;
      const answer = await unwrap(
        api.POST("/api/v1/genes/{name}/config-check", {
          params: { path: { name: sourceType } },
          body: { config, source_id: sourceId ?? null },
        }),
      );
      if (ticket === latest.current) setResult(answer);
      return answer;
    },
    [api, sourceType, sourceId],
  );
  return { result, check };
}

/** What "Check setup" found: how many items, and the first few, each with a link when the provider has one. */
export interface SetupPreview {
  count: number;
  truncated: boolean;
  items: { id: string; title: string; href: string | null }[];
}

/**
 * The identity a local folder's preview files its Documents under. An
 * existing Source has one; a new Source is given one when it is created, so
 * its preview uses a placeholder, which changes no file it lists.
 */
function previewVaultId(stored: Readonly<Record<string, unknown>>): string {
  return typeof stored.vault_id === "string" && stored.vault_id.trim() ? stored.vault_id.trim() : "preview";
}

/**
 * "Check setup": the first items the configuration would sync, read without
 * saving anything. The server reads the provider through preview-discovery,
 * with `sourceId` supplying what the form leaves out, such as a kept
 * secret; a local folder is listed by local sync on the person's computer
 * (`local_markdown_preview_tree`).
 */
export function useCheckSetup(sourceType: string, kind: CheckSetupKind | null, source?: { id: string; config: Readonly<Record<string, unknown>> }) {
  const api = useApi();
  const previewFolder = useLocalSetupJob("local_markdown_preview_tree");
  return useMutation({
    mutationFn: async (config: Record<string, unknown>): Promise<SetupPreview> => {
      if (kind === "local_folder") {
        const result = await previewFolder({
          sourceId: source?.id,
          payload: { ...config, vault_id: previewVaultId(source?.config ?? {}), limit: CHECK_SETUP_LIMIT },
        });
        return {
          count: result.counts.included,
          truncated: result.counts.included > result.items.length,
          items: result.items.map((item) => ({ id: item.relative_path, title: item.title || item.relative_path, href: null })),
        };
      }
      const preview = await unwrap(
        api.POST("/api/v1/genes/{name}/preview-discovery", {
          params: { path: { name: sourceType } },
          body: { config, limit: CHECK_SETUP_LIMIT, source_id: source?.id ?? null },
        }),
      );
      return {
        count: preview.count,
        truncated: preview.truncated,
        items: preview.items.map((item) => ({ id: item.item_id, title: item.title || item.item_id, href: item.source_url })),
      };
    },
  });
}

/** After a Source is created or saved, every list that names Sources or their projects reads again. */
function useInvalidateAfterSave() {
  const queryClient = useQueryClient();
  return () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
      queryClient.invalidateQueries({ queryKey: projectQueryKeys.all }),
      queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
    ]);
}

export function useCreateSource() {
  const api = useApi();
  const invalidate = useInvalidateAfterSave();
  return useMutation({
    mutationFn: (body: CreateSourceBody) => unwrap(api.POST("/api/v1/sources", { body })),
    onSuccess: invalidate,
  });
}

export function useUpdateSource(sourceId: string) {
  const api = useApi();
  const invalidate = useInvalidateAfterSave();
  return useMutation({
    mutationFn: (body: UpdateSourceBody) =>
      unwrap(api.PUT("/api/v1/sources/{source_id}", { params: { path: { source_id: sourceId } }, body })),
    onSuccess: invalidate,
  });
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

/** The configuration as a cache key: a secret says only whether it is typed, so a typed token never joins a key. */
function previewKey(config: Record<string, unknown>, secretKeys: readonly string[]): string {
  const secrets = new Set(secretKeys);
  return JSON.stringify(
    Object.entries(config)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => [key, secrets.has(key) ? Boolean(value) : value]),
  );
}

/**
 * What saving `config` would do to the Source's scope, asked a moment after
 * the last edit; it saves nothing. `config` is null while there is nothing to
 * ask, such as before a scope field changes. While the next answer is on its
 * way, the last one stays. A configuration the server refuses yields no
 * preview; config-check and save report it.
 */
export function useDebouncedScopeChangePreview(sourceId: string, config: Record<string, unknown> | null, secretKeys: readonly string[]) {
  const api = useApi();
  const serialized = useDebouncedValue(config ? JSON.stringify(config) : null, SCOPE_PREVIEW_DELAY_MS);
  const debounced = useMemo(() => (serialized ? (JSON.parse(serialized) as Record<string, unknown>) : null), [serialized]);
  const key = debounced ? previewKey(debounced, secretKeys) : null;
  const preview = useQuery({
    // Kept apart from the source keys, which every source write refreshes.
    queryKey: ["source-settings-scope-preview", sourceId, key] as const,
    queryFn: () =>
      unwrap(
        api.POST("/api/v1/sources/{source_id}/scope-change-preview", {
          params: { path: { source_id: sourceId } },
          body: { config: debounced! },
        }),
      ),
    enabled: key !== null,
    placeholderData: keepPreviousData,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
  return key !== null ? preview.data : undefined;
}
