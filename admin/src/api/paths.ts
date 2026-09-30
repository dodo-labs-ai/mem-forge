import type { paths } from "./schema.gen";
import type {
  LocalAgentDaemonStatus,
  LocalAgentJobCreateResponse,
  LocalAgentJobList,
  SourceListPreferences,
  SourceListResponse,
  SourceSyncReceipt,
} from "./responses";

/** Declared JSON responses for operations whose generated type is unknown. */
interface ResponsePatches {
  "/api/v1/sources": { get: SourceListResponse };
  "/api/v1/source-list/preferences": { get: SourceListPreferences; put: SourceListPreferences };
  "/api/v1/sources/{source_id}/sync": { post: SourceSyncReceipt };
  "/api/cloud/local-agent/status": { get: LocalAgentDaemonStatus };
  "/api/cloud/local-agent/jobs": { post: LocalAgentJobCreateResponse };
  "/api/cloud/local-agent/jobs/current": { get: LocalAgentJobList };
}

/** Success statuses whose generated body is replaced by the declared one. */
type SuccessStatus = 200 | 201 | 202;

type JsonResponse<T> = {
  headers: { [name: string]: unknown };
  content: { "application/json": T };
};

type PatchOperation<Operation, Body> = Operation extends { responses: infer Responses }
  ? Omit<Operation, "responses"> & { responses: Omit<Responses, SuccessStatus> & { 200: JsonResponse<Body> } }
  : Operation;

type PatchPath<PathItem, Patch> = {
  [Method in keyof PathItem]: Method extends keyof Patch ? PatchOperation<PathItem[Method], Patch[Method]> : PathItem[Method];
};

/** The generated Admin API paths with the patched responses merged in. */
export type AdminPaths = {
  [Path in keyof paths]: Path extends keyof ResponsePatches ? PatchPath<paths[Path], ResponsePatches[Path]> : paths[Path];
};
