export type ConfluenceConfigMap = Record<string, unknown>;

export interface ParsedConfluenceWikiUrl {
  normalizedBaseUrl?: string;
  apiPrefix?: string;
  spaceKey?: string;
  pageId?: string;
}

export function parseConfluenceWikiUrl(value: string): ParsedConfluenceWikiUrl {
  const text = value.trim();
  if (!text) return {};

  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return {};
  }

  const result: ParsedConfluenceWikiUrl = {
    normalizedBaseUrl: url.origin,
  };
  const pathParts = url.pathname.split("/").map((part) => part.trim()).filter(Boolean);
  let prefixParts: string[] = [];

  const spaceIndex = pathParts.indexOf("spaces");
  if (spaceIndex >= 0) {
    prefixParts = pathParts.slice(0, spaceIndex);
    const spaceKey = pathParts[spaceIndex + 1];
    if (spaceKey) result.spaceKey = decodeURIComponent(spaceKey);
    const pageId = pageIdFromPathParts(pathParts.slice(spaceIndex + 2));
    if (pageId) result.pageId = pageId;
  } else if (pathParts.length === 1) {
    prefixParts = pathParts;
  }

  const queryPageId = url.searchParams.get("pageId");
  if (queryPageId && /^\d+$/.test(queryPageId)) {
    result.pageId = queryPageId;
  }
  if (prefixParts.length > 0) {
    result.apiPrefix = `/${prefixParts.join("/")}`;
  }
  return result;
}

export function applyConfluenceUrlInference(config: ConfluenceConfigMap): ConfluenceConfigMap {
  const next = { ...config };
  const parsed = parseConfluenceWikiUrl(stringValue(next.base_url));

  if (parsed.spaceKey && listValue(next.spaces).length === 0) {
    next.spaces = [parsed.spaceKey];
  }
  if (parsed.pageId && !stringValue(next.page_tree_root)) {
    next.page_tree_root = parsed.pageId;
  }
  if (parsed.pageId) {
    next.sync_mode = "page_tree";
  } else if (parsed.spaceKey && !stringValue(next.sync_mode)) {
    next.sync_mode = "space";
  } else if (!stringValue(next.sync_mode)) {
    next.sync_mode = confluenceSyncMode(next);
  }
  return next;
}

export function confluenceSyncMode(config: ConfluenceConfigMap): "page_tree" | "space" {
  const configured = stringValue(config.sync_mode).toLowerCase();
  if (configured === "page_tree" || configured === "space") return configured;
  if (stringValue(config.page_tree_root) || parseConfluenceWikiUrl(stringValue(config.base_url)).pageId) {
    return "page_tree";
  }
  return "space";
}

export function isConfluenceFieldVisible(fieldKey: string, config: ConfluenceConfigMap): boolean {
  const mode = confluenceSyncMode(config);
  if (fieldKey === "spaces") return mode === "space";
  if (fieldKey === "page_tree_root" || fieldKey === "include_children") return mode === "page_tree";
  return true;
}

export function isConfluenceFieldRequired(fieldKey: string, config: ConfluenceConfigMap): boolean {
  const mode = confluenceSyncMode(config);
  if (fieldKey === "spaces") return mode === "space";
  if (fieldKey === "page_tree_root") return mode === "page_tree";
  return false;
}

function pageIdFromPathParts(pathParts: string[]): string | undefined {
  const pageIndex = pathParts.indexOf("pages");
  const pageId = pageIndex >= 0 ? pathParts[pageIndex + 1] : undefined;
  return pageId && /^\d+$/.test(pageId) ? pageId : undefined;
}

function listValue(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string") return value.split(",").map((item) => item.trim()).filter(Boolean);
  return [];
}

function stringValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (value == null) return "";
  return String(value);
}
