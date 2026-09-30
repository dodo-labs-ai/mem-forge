import createClient, { type Client, type Middleware } from "openapi-fetch";
import { ApiError, NoWorkspaceError, describeErrorBody } from "./errors";
import type { AdminPaths } from "./paths";
import { LOCAL_AGENT_PREFIX, RESOURCE_PREFIX, type WorkspaceController } from "./workspace";

export type ApiClient = Client<AdminPaths>;

const WORKSPACE_QUERY_PARAM = "workspace_id";

const BODYLESS_METHODS = new Set(["GET", "HEAD"]);

/** Copies a request to a new URL field by field, which every fetch implementation honors. */
async function withUrl(request: Request, url: URL): Promise<Request> {
  return new Request(url, {
    method: request.method,
    headers: request.headers,
    body: BODYLESS_METHODS.has(request.method) ? undefined : await request.arrayBuffer(),
    credentials: request.credentials,
    signal: request.signal,
  });
}

function workspaceMiddleware(workspace: WorkspaceController): Middleware {
  return {
    async onRequest({ request }) {
      const url = new URL(request.url);
      const isResource = url.pathname.startsWith(`${RESOURCE_PREFIX}/`) || url.pathname === RESOURCE_PREFIX;
      const isLocalAgent = url.pathname.startsWith(LOCAL_AGENT_PREFIX);
      if (!isResource && !isLocalAgent) return request;

      const target = workspace.current();
      if (target === null) throw new NoWorkspaceError();
      if (isResource) {
        url.pathname = target.resourceBaseUrl + url.pathname.slice(RESOURCE_PREFIX.length);
        if (target.workspaceId) url.searchParams.set(WORKSPACE_QUERY_PARAM, target.workspaceId);
      } else {
        url.pathname = target.localAgentBaseUrl + url.pathname.slice(LOCAL_AGENT_PREFIX.length);
      }
      return withUrl(request, url);
    },
  };
}

/** Creates the typed Admin API client bound to the current workspace target. */
export function createApiClient(workspace: WorkspaceController, origin = window.location.origin): ApiClient {
  const client = createClient<AdminPaths>({ baseUrl: origin });
  client.use(workspaceMiddleware(workspace));
  return client;
}

interface FetchResult<T> {
  data?: T;
  error?: unknown;
  response: Response;
}

/** Returns the response data, or throws an `ApiError` carrying the server's message. */
export async function unwrap<T>(call: Promise<FetchResult<T>>): Promise<T> {
  const { data, error, response } = await call;
  if (!response.ok) throw new ApiError(response.status, describeErrorBody(error, response.status));
  return data as T;
}
