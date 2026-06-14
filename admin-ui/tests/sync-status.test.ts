import assert from "node:assert/strict";

import { buildFailureDetails } from "../src/components/admin/syncFailureDetails.js";

const embeddingDetails = buildFailureDetails({
  failed_docs: [
    {
      doc_id: "doc-embedding",
      title: "Payroll_run_request_stuck",
      error: "Embedding provider unreachable: [Errno 111] Connection refused",
    },
  ],
});

assert.equal(embeddingDetails?.groups[0]?.label, "Embedding provider unreachable");
assert.equal(
  embeddingDetails?.groups[0]?.help,
  "MemForge could not reach the configured embedding provider. Check the provider endpoint, network access, and service status, then retry the sync.",
);

const timeoutDetails = buildFailureDetails({
  failed_docs: [
    {
      doc_id: "doc-timeout",
      title: "Payroll_timeout",
      error: "litellm.APIConnectionError: All connection attempts failed",
    },
  ],
});

assert.equal(timeoutDetails?.groups[0]?.label, "LLM provider unreachable");

const llmDetails = buildFailureDetails({
  failed_docs: [
    {
      doc_id: "doc-llm",
      title: "Payroll_run_not_getting_triggered",
      error: "litellm.InternalServerError: AnthropicException - Cannot connect to host provider.example:443",
    },
  ],
});

assert.equal(llmDetails?.groups[0]?.label, "LLM provider unreachable");
assert.equal(
  llmDetails?.groups[0]?.help,
  "MemForge could not reach the configured LLM provider. Check the provider endpoint, network access, and service status, then retry the sync.",
);

const rateLimitDetails = buildFailureDetails({
  failed_docs: [
    {
      doc_id: "doc-rate-limit",
      title: "Confluence_rate_limit",
      error: "Confluence returned 429 rate limit",
    },
  ],
});

assert.equal(rateLimitDetails?.groups[0]?.label, "Rate limited by Confluence");

const certificateDetails = buildFailureDetails({
  failed_docs: [
    {
      doc_id: "doc-certificate",
      title: "Confluence_certificate",
      error: "certificate_verify_failed",
    },
  ],
});

assert.equal(certificateDetails?.groups[0]?.label, "Certificate verification failed");
