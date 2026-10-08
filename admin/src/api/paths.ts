import type { paths } from "./schema.gen";
import type { LocalAgentJob } from "./responses";

/**
 * JSON responses the UI reads more precisely than the generated schema
 * states. Keep this list short: a route belongs here only while its schema
 * leaves part of the body free-form.
 */
interface ResponsePatches {
  "/api/cloud/local-agent/jobs/current": { get: { data: LocalAgentJob[] } };
  "/api/cloud/local-agent/jobs/{job_id}": { get: LocalAgentJob };
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
