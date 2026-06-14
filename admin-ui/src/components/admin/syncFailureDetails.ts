import type { SyncStatus } from "../../api/types.js";

type SyncFailureDoc = NonNullable<SyncStatus["failed_docs"]>[number];
type SyncFailureStatus = Pick<SyncStatus, "failed_docs">;

export type FailureGroup = {
  label: string;
  help: string;
  items: SyncFailureDoc[];
};

export function buildFailureDetails(sync: SyncFailureStatus) {
  const failedDocs = sync.failed_docs ?? [];
  if (failedDocs.length === 0) return null;

  const groups = new Map<string, FailureGroup>();
  for (const doc of failedDocs) {
    const key = failureCategory(doc.error);
    if (!groups.has(key)) {
      groups.set(key, failureGroup(key));
    }
    groups.get(key)?.items.push(doc);
  }

  return { groups: Array.from(groups.values()) };
}

function failureCategory(error: string) {
  const normalized = error.toLowerCase();
  if (normalized.includes("embedding provider unreachable")) return "embedding_provider_unreachable";
  if (normalized.includes("llm provider unreachable")) return "llm_provider_unreachable";
  if (
    isProviderConnectivityError(normalized) &&
    (normalized.includes("litellm") ||
      normalized.includes("anthropicexception") ||
      normalized.includes("openaiexception"))
  ) {
    return "llm_provider_unreachable";
  }
  if (normalized.includes("rate limit") || normalized.includes("429")) return "rate_limit";
  if (normalized.includes("pdf export") || normalized.includes("did not produce a pdf")) return "pdf_export";
  if (normalized.includes("certificate_verify_failed") || normalized.includes("certificate verify")) return "certificate";
  return "other";
}

function isProviderConnectivityError(normalized: string) {
  // Keep this list aligned with src/memforge/pipeline/sync.py.
  return [
    "all connection attempts failed",
    "cannot connect to host",
    "connect call failed",
    "connect timeout",
    "connection refused",
    "connection timed out",
    "failed to connect",
    "name or service not known",
    "network is unreachable",
    "no route to host",
    "nodename nor servname",
    "temporary failure in name resolution",
  ].some((marker) => normalized.includes(marker));
}

function failureGroup(key: string): FailureGroup {
  if (key === "embedding_provider_unreachable") {
    return {
      label: "Embedding provider unreachable",
      help: "MemForge could not reach the configured embedding provider. Check the provider endpoint, network access, and service status, then retry the sync.",
      items: [],
    };
  }
  if (key === "llm_provider_unreachable") {
    return {
      label: "LLM provider unreachable",
      help: "MemForge could not reach the configured LLM provider. Check the provider endpoint, network access, and service status, then retry the sync.",
      items: [],
    };
  }
  if (key === "rate_limit") {
    return {
      label: "Rate limited by Confluence",
      help: "Confluence temporarily limited export requests. Wait a few minutes, then retry the sync.",
      items: [],
    };
  }
  if (key === "pdf_export") {
    return {
      label: "PDF export unavailable",
      help: "Confluence did not return a usable PDF for these documents.",
      items: [],
    };
  }
  if (key === "certificate") {
    return {
      label: "Certificate verification failed",
      help: "The local Python runtime could not verify the Confluence certificate chain.",
      items: [],
    };
  }
  return {
    label: "Other sync errors",
    help: "These documents failed for another reason.",
    items: [],
  };
}
