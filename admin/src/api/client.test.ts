import { afterEach, describe, expect, test, vi } from "vitest";
import { fakeFetch } from "@/test/fakeApi";
import { createApiClient, unwrap, workspaceResourceUrl } from "./client";
import { ApiError, NoWorkspaceError, isReindexInProgress, isSearchIndexRefusal } from "./errors";
import { STANDALONE_TARGET, createWorkspaceController } from "./workspace";

const ORIGIN = "http://admin.test";

describe("workspaceResourceUrl", () => {
  test("routes links and preserves queries and fragments without a duplicate selector", () => {
    const target = { ...STANDALONE_TARGET, resourceBaseUrl: "/api/cloud/workspaces/ws-1/v1", workspaceId: "ws-1" };
    const url = new URL(workspaceResourceUrl("/api/v1/source-units/unit-1/content?view=raw&workspace_id=old#evidence", target, ORIGIN));
    expect(url.pathname).toBe("/api/cloud/workspaces/ws-1/v1/source-units/unit-1/content");
    expect(url.searchParams.getAll("workspace_id")).toEqual(["ws-1"]);
    expect(url.searchParams.get("view")).toBe("raw");
    expect(url.hash).toBe("#evidence");
  });

  test("does not send workspace selection to external sources or host routes", () => {
    for (const href of ["https://wiki.example.test/api/v1/page", "/api/cloud/session", "/api/v10/content"]) {
      expect(workspaceResourceUrl(href, STANDALONE_TARGET, ORIGIN)).toBe(href);
    }
  });

  test("requires a selected workspace for resource links", () => {
    expect(() => workspaceResourceUrl("/api/v1/source-units/unit-1/pdf", null, ORIGIN)).toThrow(NoWorkspaceError);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createApiClient", () => {
  test("sends resource routes to the workspace with its id", async () => {
    const { calls } = fakeFetch({ "GET /api/cloud/workspaces/ws-1/v1/sources": () => ({ data: [] }) });
    const workspace = createWorkspaceController(() => {});
    workspace.setTarget({
      resourceBaseUrl: "/api/cloud/workspaces/ws-1/v1/",
      localAgentBaseUrl: "/api/cloud/workspaces/ws-1/local-agent",
      workspaceId: "ws-1",
      shared: true,
    });
    const api = createApiClient(workspace, ORIGIN);

    await unwrap(api.GET("/api/v1/sources"));

    expect(new URL(calls[0]!.url).searchParams.get("workspace_id")).toBe("ws-1");
  });

  test("sends local sync routes to the workspace's local sync base", async () => {
    const { calls } = fakeFetch({
      "GET /api/cloud/workspaces/ws-1/local-agent/status": () => ({ status: "online" }),
    });
    const workspace = createWorkspaceController(() => {});
    workspace.setTarget({ ...STANDALONE_TARGET, localAgentBaseUrl: "/api/cloud/workspaces/ws-1/local-agent", workspaceId: "ws-1" });
    const api = createApiClient(workspace, ORIGIN);

    await unwrap(api.GET("/api/cloud/local-agent/status"));

    expect(new URL(calls[0]!.url).searchParams.has("workspace_id")).toBe(false);
  });

  test("refuses requests while no workspace is selected", async () => {
    fakeFetch({});
    const workspace = createWorkspaceController(() => {}, null);
    const api = createApiClient(workspace, ORIGIN);

    await expect(api.GET("/api/v1/sources")).rejects.toBeInstanceOf(NoWorkspaceError);
  });

  test("reads an admitted local job at the host route after workspace selection changes", async () => {
    const { calls } = fakeFetch({ "GET /api/cloud/local-agent/jobs/laj-auth": () => ({ status: "succeeded" }) });
    const workspace = createWorkspaceController(() => {}, {
      ...STANDALONE_TARGET,
      localAgentBaseUrl: "/api/cloud/workspaces/ws-1/local-agent",
      workspaceId: "ws-1",
      shared: true,
    });
    const api = createApiClient(workspace, ORIGIN);
    await unwrap(api.GET("/api/cloud/local-agent/jobs/{job_id}", { params: { path: { job_id: "laj-auth" } } }));
    workspace.setTarget(null);
    await unwrap(api.GET("/api/cloud/local-agent/jobs/{job_id}", { params: { path: { job_id: "laj-auth" } } }));
    expect(calls.map((request) => new URL(request.url).pathname)).toEqual([
      "/api/cloud/local-agent/jobs/laj-auth", "/api/cloud/local-agent/jobs/laj-auth",
    ]);
  });

  test("clears cached data only when the target really changes", () => {
    const onChange = vi.fn();
    const workspace = createWorkspaceController(onChange);
    workspace.setTarget({ ...STANDALONE_TARGET, resourceBaseUrl: "/api/v1/" });
    expect(onChange).not.toHaveBeenCalled();
    workspace.setTarget({ ...STANDALONE_TARGET, workspaceId: "other" });
    expect(onChange).toHaveBeenCalledTimes(1);
  });
});

describe("unwrap", () => {
  test("raises the server's explanation", async () => {
    const response = new Response(null, { status: 409 });
    await expect(unwrap(Promise.resolve({ error: { detail: "Source is syncing" }, response }))).rejects.toEqual(
      new ApiError(409, "Source is syncing"),
    );
  });

  test("reads the message of a structured refusal", async () => {
    const response = new Response(null, { status: 403 });
    const error = { detail: { error: "project_management_forbidden", message: "Only a workspace admin can change projects." } };
    await expect(unwrap(Promise.resolve({ error, response }))).rejects.toEqual(
      new ApiError(403, "Only a workspace admin can change projects.", "project_management_forbidden"),
    );
  });

  test("explains a change refused while the search index is rebuilt, and keeps its code", async () => {
    const response = new Response(null, { status: 409 });
    const error = {
      detail: {
        error: "reindex_in_progress",
        message: "The Memory vector index is being rebuilt. Memories cannot be changed until the rebuild has finished.",
      },
    };
    const refused = await unwrap(Promise.resolve({ error, response })).catch((caught: unknown) => caught);

    expect(refused).toEqual(
      new ApiError(
        409,
        "The search index is being rebuilt, so this change has to wait. Try again when the rebuild has finished. Settings shows its progress.",
        "reindex_in_progress",
      ),
    );
    expect(isReindexInProgress(refused)).toBe(true);
    expect(isSearchIndexRefusal(refused)).toBe(true);
  });

  test("a change that waits for an index of the model in use keeps the server's words, and is known by its code", async () => {
    const message = "The vector index has to be rebuilt for the embedding model in use before this change can be made.";

    for (const code of ["reindex_required", "vector_index_missing"]) {
      const response = new Response(null, { status: 409 });
      const refused = await unwrap(Promise.resolve({ error: { detail: { error: code, message } }, response })).catch(
        (caught: unknown) => caught,
      );

      expect(refused).toEqual(new ApiError(409, message, code));
      expect(isSearchIndexRefusal(refused)).toBe(true);
      expect(isReindexInProgress(refused)).toBe(false);
    }
  });

  test("a failure that is not about the search index is not taken for one", async () => {
    const response = new Response(null, { status: 409 });
    const changed = await unwrap(Promise.resolve({ error: { detail: "Review participants changed" }, response })).catch(
      (error: unknown) => error,
    );

    expect(changed).toMatchObject({ code: null, message: "Review participants changed" });
    expect(isSearchIndexRefusal(changed)).toBe(false);
    expect(isSearchIndexRefusal(new Error("Network down"))).toBe(false);
  });

  test("joins validation messages", async () => {
    const response = new Response(null, { status: 422 });
    const error = { detail: [{ msg: "Field required" }, { msg: "Too long" }] };
    await expect(unwrap(Promise.resolve({ error, response }))).rejects.toThrow("Field required Too long");
  });
});
