import { errorMessage } from "@/lib/errors";

/** Reason codes the memory actions return, in words the user can act on. */
const ACTION_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  content_hash_mismatch: "Someone changed this memory after you opened it. Reload to see the latest version before you try again.",
  memory_content_changed: "One of these memories changed after you opened it. Reload to see the latest version before you try again.",
  relation_label_changed: "This relation changed after you opened it. Reload to see the current relation.",
  memory_not_active: "This memory is no longer active. Reload to see its current state.",
  source_backed_memory_requires_lifecycle_review:
    "This memory is backed by a source. Remove it from the source, or propose a correction.",
  memory_owner_authority_required: "Only the owner of this private memory can change it.",
  workspace_memory_correction_requires_management_authority:
    "No source backs this memory, so only someone who manages workspace memories can correct it.",
};

/** The message for a failed memory action, translating the server's reason codes. */
export function actionErrorMessage(error: unknown): string {
  const message = errorMessage(error);
  return ACTION_ERROR_MESSAGES[message] ?? message;
}
