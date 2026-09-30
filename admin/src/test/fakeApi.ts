import { vi } from "vitest";

export type Route = (request: Request) => unknown;

/**
 * Replaces `fetch` with a router keyed by "METHOD /path". A route returns the
 * JSON body of a 200 response, or a `Response` to answer with another status.
 * Unmatched requests fail the test, so a page cannot silently depend on an
 * endpoint.
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
    if (body instanceof Response) return body;
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** A JSON error response with FastAPI's `detail` body. */
export function errorResponse(status: number, detail: unknown): Response {
  return new Response(JSON.stringify({ detail }), { status, headers: { "Content-Type": "application/json" } });
}
