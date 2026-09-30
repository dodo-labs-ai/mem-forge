/**
 * Reserved project keys are system buckets the resolver writes to and the
 * UI hides from regular pickers. They are imported from one place so a
 * future bucket only requires editing this file.
 */
export const SHARED_PROJECT_KEY = "SHARED";
export const UNSORTED_PROJECT_KEY = "UNSORTED";
export const RESERVED_PROJECT_KEYS: readonly string[] = [
  SHARED_PROJECT_KEY,
  UNSORTED_PROJECT_KEY,
];

export function isReservedProjectKey(key: string): boolean {
  return RESERVED_PROJECT_KEYS.includes(key);
}
