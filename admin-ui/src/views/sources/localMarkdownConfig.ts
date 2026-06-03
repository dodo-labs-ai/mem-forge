export function buildLocalMarkdownPushCommand({
  vaultId,
  sourceId,
}: {
  vaultId: string;
  sourceId: string | null;
}): string {
  const vault = vaultId.trim() || "<vault-id>";
  const source = (sourceId ?? "").trim() || "<source-id>";
  return `memforge adapter kb push ${vault} --source-id ${source}`;
}
