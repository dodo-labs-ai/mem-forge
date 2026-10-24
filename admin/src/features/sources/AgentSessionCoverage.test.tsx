import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import { errorResponse } from "@/test/fakeApi";
import { renderWithApi } from "@/test/renderWithApi";
import { makeSource } from "@/test/sourceFixtures";
import { AgentSessionCoverage } from "./AgentSessionCoverage";

const SOURCE = makeSource({ id: "src-codex", type: "agent_session", client: "codex", name: "Codex Session" });
const PROJECTS = { data: [{ id: "p-app", key: "APP", name: "Payroll app", kind: "normal", memory_count: 9, created_at: null }], can_manage: true };

function completeness(overrides: Record<string, unknown> = {}) {
  return {
    session_id: null,
    source_id: "src-codex",
    total: 13,
    processed_total: 12,
    counts: { knowledge_patched: 4, no_output: 8, failed: 1 },
    no_output_fraction: 8 / 12,
    latest_failure: { count: 1, reason: "The model timed out.", last_seen_at: "2026-10-24T08:00:00Z" },
    ...overrides,
  };
}

test("shows how the session windows turned out and the projects their memories went to", async () => {
  const { calls } = renderWithApi(<AgentSessionCoverage source={SOURCE} />, {
    "GET /api/v1/agent-sessions/completeness": () => completeness(),
    "GET /api/v1/sources/src-codex/projects": () => ({
      source_id: "src-codex",
      projects: [
        { project: "APP", document_count: 3, memory_count: 7, last_observed_at: "2026-10-23T08:00:00Z" },
        { project: "UNSORTED", document_count: 1, memory_count: 1, last_observed_at: null },
      ],
    }),
    "GET /api/v1/projects": () => PROJECTS,
  });

  expect(await screen.findByText("Windows processed")).toBeInTheDocument();
  expect(screen.getByText("12")).toBeInTheDocument();
  expect(screen.getByText("Kept as memory").nextSibling).toHaveTextContent("4");
  expect(screen.getByText("Skipped as low-signal").nextSibling).toHaveTextContent("8");
  expect(screen.getByText("1 window failed")).toBeInTheDocument();
  expect(screen.getByText(/The model timed out\./)).toBeInTheDocument();

  const projects = await screen.findByRole("list", { name: "Projects with memories from sessions" });
  const rows = within(projects).getAllByRole("listitem");
  expect(rows).toHaveLength(2);
  expect(await within(rows[0]!).findByText("Payroll app")).toBeInTheDocument();
  expect(within(rows[0]!).getByText("3 summaries")).toBeInTheDocument();
  expect(within(rows[0]!).getByText("7 memories")).toBeInTheDocument();
  expect(within(rows[1]!).getByText("UNSORTED")).toBeInTheDocument();
  const completenessRequest = calls.find((request) => new URL(request.url).pathname === "/api/v1/agent-sessions/completeness")!;
  expect(new URL(completenessRequest.url).searchParams.get("source_id")).toBe("src-codex");
});

test("a source no window has reached yet says so and warns about nothing", async () => {
  renderWithApi(<AgentSessionCoverage source={SOURCE} />, {
    "GET /api/v1/agent-sessions/completeness": () =>
      completeness({ total: 0, processed_total: 0, counts: { knowledge_patched: 0, no_output: 0, failed: 0 }, latest_failure: null }),
    "GET /api/v1/sources/src-codex/projects": () => ({ source_id: "src-codex", projects: [] }),
    "GET /api/v1/projects": () => PROJECTS,
  });

  expect(await screen.findByText("No session has been kept yet.")).toBeInTheDocument();
  expect(await screen.findByText("Windows processed")).toBeInTheDocument();
  expect(screen.queryByText(/failed/)).not.toBeInTheDocument();
});

test("a coverage read that fails can be retried", async () => {
  let reads = 0;
  renderWithApi(<AgentSessionCoverage source={SOURCE} />, {
    "GET /api/v1/agent-sessions/completeness": () => (++reads === 1 ? errorResponse(503, "Service unavailable") : completeness()),
    "GET /api/v1/sources/src-codex/projects": () => ({ source_id: "src-codex", projects: [] }),
    "GET /api/v1/projects": () => PROJECTS,
  });

  expect(await screen.findByText("Could not load session coverage")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /Retry|Try again/ }));
  expect(await screen.findByText("Windows processed")).toBeInTheDocument();
});
