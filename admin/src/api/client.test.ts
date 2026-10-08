import { afterEach, describe, expect, test, vi } from "vitest";
import { fakeFetch } from "@/test/fakeApi";
import { createApiClient, unwrap, workspaceResourceUrl } from "./client";
import { ApiError, NoWorkspaceError } from "./errors";
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
      new ApiError(403, "Only a workspace admin can change projects."),
    );
  });

  test("joins validation messages", async () => {
    const response = new Response(null, { status: 422 });
    const error = { detail: [{ msg: "Field required" }, { msg: "Too long" }] };
    await expect(unwrap(Promise.resolve({ error, response }))).rejects.toThrow("Field required Too long");
  });
});
