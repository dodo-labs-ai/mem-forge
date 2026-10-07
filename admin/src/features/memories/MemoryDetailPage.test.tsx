import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { errorResponse } from "@/test/fakeApi";
import { makeMemoryDetail, makeRelation } from "@/test/memoryFixtures";
import { memoriesRoutes, renderMemories } from "@/test/renderMemories";
import type { MemoryDetail } from "./model/types";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderDetail(memory: MemoryDetail, extraRoutes = {}) {
  return renderMemories(
    `/memories/${memory.id}`,
    memoriesRoutes({ [`GET /api/v1/memories/${memory.id}`]: () => memory, ...extraRoutes }),
  );
}

test("shows the memory's details, relations and evidence", async () => {
  renderDetail(makeMemoryDetail());

  expect(await screen.findByRole("heading", { level: 1, name: /Cut-off runs the same way/ })).toBeInTheDocument();
  const details = screen.getByRole("region", { name: "Details" });
  expect(within(details).getByText("Everyone in Mount Tai")).toBeInTheDocument();
  expect(within(details).getByText("payroll-v2")).toBeInTheDocument();
  expect(within(details).getByText("Payroll handbook")).toBeInTheDocument();

  const entities = screen.getByRole("region", { name: "Entities" });
  expect(within(entities).getByText("Cut-off")).toBeInTheDocument();
  expect(within(entities).queryByRole("link")).not.toBeInTheDocument();

  const related = screen.getByRole("region", { name: "Related memories in other documents" });
  expect(within(related).getByText("Conflicts with")).toBeInTheDocument();
  expect(within(related).getByRole("link", { name: /on-demand lifecycle does not support/ })).toHaveAttribute(
    "href",
    "/memories/mem-ondemand",
  );

  const evidence = screen.getByRole("region", { name: "Evidence" });
  expect(within(evidence).getByText("Payroll Handbook")).toBeInTheDocument();
  expect(within(evidence).getByText("Primary")).toBeInTheDocument();
  expect(within(evidence).getByRole("link", { name: "Open content" })).toBeInTheDocument();
});

test("shows focused evidence text and keeps the complete source available on expansion", async () => {
  const user = userEvent.setup();
  const memory = makeMemoryDetail();
  const item = memory.evidence![0]!.items[0]!;
  item.text = "Cut-off applies uniformly to the three lifecycle types.";
  renderDetail(memory);
  const evidence = await screen.findByRole("region", { name: "Evidence" });
  expect(within(evidence).getByText(item.text)).toBeVisible();
  const source = within(evidence).getByText(item.excerpt!);
  expect(source).not.toBeVisible();
  await user.click(within(evidence).getByText("Source evidence"));
  expect(source).toBeVisible();
  expect(evidence).not.toHaveTextContent("LLM");
});

test("a memory that waits for review opens the review it waits for", async () => {
  const { calls } = renderDetail(makeMemoryDetail({ status: "pending_review", open_review_id: "rev-pending" }));

  expect(await screen.findByRole("link", { name: "Open review" })).toHaveAttribute("href", "/review/rev-pending");
  expect(calls.some((request) => new URL(request.url).pathname === "/api/v1/memory-reviews")).toBe(false);
});

test("a source-backed memory offers Retire only with the reason it cannot", async () => {
  const user = userEvent.setup();
  renderDetail(makeMemoryDetail({ source_backed: true }));

  await user.click(await screen.findByRole("button", { name: "More actions" }));
  const retire = await screen.findByRole("menuitem", { name: /Retire memory/ });
  expect(retire).toHaveAttribute("aria-disabled", "true");
  expect(retire).toHaveTextContent("This memory is backed by a source. Remove it from the source, or propose a correction.");
});

test("retiring asks for a reason, guards against a changed memory, then shows the retired memory", async () => {
  const user = userEvent.setup();
  const memory = makeMemoryDetail({ source_backed: false });
  let current = memory;
  const { calls } = renderDetail(memory, {
    [`GET /api/v1/memories/${memory.id}`]: () => current,
    [`POST /api/v1/memories/${memory.id}/retire`]: () => {
      current = { ...memory, status: "retired", retirement_reason: "No longer true after the workflow change" };
      return { memory_id: memory.id, status: "retired" };
    },
  });

  await user.click(await screen.findByRole("button", { name: "More actions" }));
  await user.click(await screen.findByRole("menuitem", { name: /Retire memory/ }));
  const dialog = await screen.findByRole("dialog", { name: "Retire this memory?" });
  await user.click(within(dialog).getByRole("button", { name: "Retire memory" }));
  expect(await within(dialog).findByText("Enter why this memory should be retired.")).toBeInTheDocument();

  await user.type(within(dialog).getByLabelText(/Reason/), "No longer true after the workflow change");
  await user.click(within(dialog).getByRole("button", { name: "Retire memory" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const retire = calls.find((request) => request.method === "POST")!;
  expect(await retire.clone().json()).toEqual({
    reason: "No longer true after the workflow change",
    expected_content_hash: memory.content_hash,
  });
  expect(await screen.findByText("Retired.")).toBeInTheDocument();
});

test("dismissing a relation asks first and sends the note", async () => {
  const user = userEvent.setup();
  const memory = makeMemoryDetail({ relations: [makeRelation()] });
  const { calls } = renderDetail(memory, {
    [`POST /api/v1/memories/${memory.id}/relations/mem-ondemand/dismissal`]: () => ({
      dismissal_id: "dis-1",
      memory_low_id: "mem-cutoff",
      memory_high_id: "mem-ondemand",
      label: "contradicts",
      dismissed_by: "local",
      dismissed_at: "2026-09-30T10:00:00Z",
    }),
  });

  await user.click(await screen.findByRole("button", { name: "Not a conflict" }));
  const dialog = await screen.findByRole("dialog", { name: "Mark as not a conflict?" });
  await user.type(within(dialog).getByLabelText("Note (optional)"), "Different lifecycles");
  await user.click(within(dialog).getByRole("button", { name: "Not a conflict" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const dismissal = calls.find((request) => request.method === "POST")!;
  expect(await dismissal.clone().json()).toEqual({
    label: "contradicts",
    expected_content_hash: memory.content_hash,
    counterpart_expected_content_hash: "hash-ondemand",
    note: "Different lifecycles",
  });
});

test("a superseded memory explains its status and links the newer memory", async () => {
  const memory = makeMemoryDetail({
    status: "superseded",
    superseded_at: "2026-09-26T11:40:00Z",
    superseded_by: "mem-newer",
    replacement_reason: "Replaced by a newer decision",
  });
  renderDetail(memory, {
    "GET /api/v1/memories/mem-newer": () => makeMemoryDetail({ id: "mem-newer", content: "Per-row acceptance keeps valid rows." }),
  });

  const banner = await screen.findByRole("status");
  expect(within(banner).getByText("Superseded.")).toBeInTheDocument();
  expect(banner).toHaveTextContent("A newer memory replaced this one on Sep 26, 2026.");
  expect(within(banner).getByRole("link", { name: "Open newer memory" })).toHaveAttribute("href", "/memories/mem-newer");
  expect(screen.queryByRole("button", { name: "Propose correction" })).not.toBeInTheDocument();

  const lifecycle = screen.getByRole("region", { name: "Lifecycle" });
  expect(await within(lifecycle).findByRole("link", { name: "Per-row acceptance keeps valid rows." })).toBeInTheDocument();
});

test("a memory the caller cannot see reads as not found", async () => {
  renderMemories(
    "/memories/mem-hidden",
    memoriesRoutes({
      "GET /api/v1/memories/mem-hidden": () => errorResponse(404, "Memory not found"),
    }),
  );

  expect(await screen.findByText("Memory not found")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Back to Memories" })).toHaveAttribute("href", "/memories");
});
