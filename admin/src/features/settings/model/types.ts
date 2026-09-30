import type { components } from "@/api";

type Schemas = components["schemas"];

export type LlmConfig = Schemas["LlmConfigResponse"];
export type LlmConfigUpdate = Schemas["LlmConfigRequest"];
export type LlmProbeRequest = Schemas["LlmConfigProbeRequest"];
export type LlmProbeResult = Schemas["LlmConfigProbeResponse"];

/** The two model endpoints MemForge calls: one generates text, one embeds it. */
export type EndpointKind = LlmProbeRequest["kind"];
