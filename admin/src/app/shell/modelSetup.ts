import type { ModelSettings } from "@/api";
import { findReservedRedirect, isProductPath, type AdminExtension } from "../extension/contract";
import { SETTINGS_NAV_ITEM } from "../navigation";

const SETTINGS_PATH = `/${SETTINGS_NAV_ITEM.segment}`;

export interface ModelSetupMessage {
  title: string;
  body: string;
}

/**
 * What to tell the user while no generation model is ready, or `null` when
 * one is. Without a generation model a source sync writes no memories, so the
 * user should hear it wherever they are, not only on the Settings page.
 */
export function modelSetupMessage(settings: ModelSettings): ModelSetupMessage | null {
  const model = settings.generation.active;
  if (model === null) {
    return {
      title: "No generation model is set up",
      body: settings.writable
        ? "MemForge cannot write memories from your sources until you choose one."
        : "MemForge cannot write memories from your sources until the deployment environment names one.",
    };
  }
  if (model.state === "ready") return null;
  if (model.validation === null) {
    return {
      title: "The generation model has not been validated yet",
      body: "MemForge uses it as it is, and validates it in the background after the server starts.",
    };
  }
  if (!model.validation.passed) {
    return {
      title: "The generation model failed its latest validation",
      body: "It stays in use, so writing memories may fail until this is fixed. Settings shows the reason.",
    };
  }
  return {
    title: "The generation model needs revalidation",
    body: "Its settings or the installed model library changed since it last passed. Validate it again in Settings.",
  };
}

/**
 * What to tell the user while vector search is off because the search index
 * has to be rebuilt for the embedding model in use, or is being rebuilt
 * after it had to be; `null` otherwise. Search still answers, by matching
 * words, so nothing on a page says that its results are weaker than they
 * could be.
 */
export function searchIndexMessage(settings: ModelSettings): ModelSetupMessage | null {
  const { status, serves_search: servesSearch } = settings.vector_index;
  if (status === "reindex_required") {
    return {
      title: "Vector search is off",
      body: "The search index does not fit the embedding model in use. Search ranks by matching words until the index is rebuilt in Settings.",
    };
  }
  if (status === "reindexing" && !servesSearch) {
    return {
      title: "Vector search is off",
      body: "The search index is being rebuilt for the embedding model in use. Search ranks by matching words until the rebuild has finished.",
    };
  }
  return null;
}

/** Everything the notice has to say about the models, the generation model first. */
export function modelSetupMessages(settings: ModelSettings): ModelSetupMessage[] {
  return [modelSetupMessage(settings), searchIndexMessage(settings)].filter((message) => message !== null);
}

/**
 * Whether `pathname` is a page the notices belong on: every product page but
 * Settings, which shows the same state in full. Extension pages are not about
 * the selected workspace, whose models these are.
 */
export function showsModelSetup(pathname: string): boolean {
  const onSettings = pathname === SETTINGS_PATH || pathname.startsWith(`${SETTINGS_PATH}/`);
  return isProductPath(pathname) && !onSettings;
}

/**
 * Where a notice's link to Settings goes: the extension's own settings page
 * when it redirects Settings, and nowhere for a user who cannot open that page.
 */
export function settingsLinkTarget(extension: AdminExtension | undefined): string | null {
  const redirect = findReservedRedirect(extension, SETTINGS_NAV_ITEM.segment);
  if (redirect === undefined) return SETTINGS_PATH;
  return (redirect.visibleWhen?.() ?? true) ? redirect.to : null;
}
