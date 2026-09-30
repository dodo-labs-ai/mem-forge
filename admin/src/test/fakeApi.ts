import { vi } from "vitest";

export type Route = (request: Request) => unknown;

/**
 * Replaces `fetch` with a router keyed by "METHOD /path". Unmatched requests
 * fail the test, so a page cannot silently depend on an endpoint.
 */
export function fakeFetch(routes: Record<string, Route>) {
  const calls: Request[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init);
    calls.push(request);
    const key = `${request.method} ${new URL(request.url).pathname}`;
    const route = routes[key];
    if (!route) throw new Error(`Unexpected request: ${key}`);
    const body = await route(request);
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}
