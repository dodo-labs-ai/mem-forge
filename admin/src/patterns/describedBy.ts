/** The `aria-describedby` value for the ids that are present. */
export function describedBy(...ids: Array<string | false | null | undefined>): string | undefined {
  const present = ids.filter(Boolean);
  return present.length > 0 ? present.join(" ") : undefined;
}
