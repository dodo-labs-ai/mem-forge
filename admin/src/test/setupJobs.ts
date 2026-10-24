import type { SetupJobOperation } from "@/api";
import type { Route } from "./fakeApi";

/** How a fake daemon completes one setup operation: the job it reports, as the host job route returns it. */
export type FakeSetupJob = { status: "succeeded" | "failed"; result: Record<string, unknown>; last_error?: string | null };

/**
 * Fake API routes for local setup jobs: local sync is online, creating a job
 * answers `laj-<operation>`, and reading it answers what `jobs` gives for that
 * operation, a job or a function of how often it was read. Each created job's
 * request body is pushed to `created`.
 */
export function setupJobRoutes(
  jobs: Partial<Record<SetupJobOperation, FakeSetupJob | ((reads: number) => FakeSetupJob | Response)>>,
  created: { operation: SetupJobOperation; source_id: string; payload: Record<string, unknown> }[] = [],
): Record<string, Route> {
  const routes: Record<string, Route> = {
    "GET /api/cloud/local-agent/status": () => ({ status: "online" }),
    "POST /api/cloud/local-agent/jobs": async (request) => {
      const body = (await request.clone().json()) as (typeof created)[number];
      created.push(body);
      return { job_id: `laj-${body.operation}`, status: "queued" };
    },
  };
  for (const [operation, job] of Object.entries(jobs)) {
    let reads = 0;
    routes[`GET /api/cloud/local-agent/jobs/laj-${operation}`] = () => {
      reads += 1;
      const answer = typeof job === "function" ? job(reads) : job;
      return answer instanceof Response ? answer : { job_id: `laj-${operation}`, last_error: null, ...answer };
    };
  }
  return routes;
}
