/** A short excerpt of one side of a change, split around the words that differ. */
export interface ChangeCueExcerpt {
  before: string;
  changed: string;
  after: string;
}

export interface ChangeCue {
  current: ChangeCueExcerpt | null;
  proposed: ChangeCueExcerpt | null;
}

/** Unchanged words kept on each side of the change for context. */
const CONTEXT_WORDS = 4;
/** Changed words shown before the excerpt is cut. */
const CHANGED_WORDS = 8;
const ELLIPSIS = "…";

function words(value: string | null | undefined): string[] {
  return value?.trim().split(/\s+/).filter(Boolean) ?? [];
}

function excerpt(tokens: string[], start: number, end: number): ChangeCueExcerpt | null {
  if (tokens.length === 0) return null;
  const changedEnd = Math.min(end, start + CHANGED_WORDS);
  const beforeStart = Math.max(0, start - CONTEXT_WORDS);
  const afterEnd = Math.min(tokens.length, changedEnd + CONTEXT_WORDS);
  return {
    before: `${beforeStart > 0 ? `${ELLIPSIS} ` : ""}${tokens.slice(beforeStart, start).join(" ")}`,
    changed: tokens.slice(start, changedEnd).join(" "),
    after: `${tokens.slice(changedEnd, afterEnd).join(" ")}${afterEnd < tokens.length ? ` ${ELLIPSIS}` : ""}`,
  };
}

function plainExcerpt(tokens: string[]): ChangeCueExcerpt | null {
  if (tokens.length === 0) return null;
  const end = Math.min(tokens.length, CHANGED_WORDS + CONTEXT_WORDS);
  return { before: `${tokens.slice(0, end).join(" ")}${end < tokens.length ? ` ${ELLIPSIS}` : ""}`, changed: "", after: "" };
}

/**
 * Finds the run of words that differs between the current and the proposed
 * text, so a queue row can highlight what the decision changes. When one side
 * is missing or both are equal, each side is a plain excerpt.
 */
export function buildChangeCue(current: string | null | undefined, proposed: string | null | undefined): ChangeCue {
  const currentWords = words(current);
  const proposedWords = words(proposed);
  if (currentWords.length === 0 || proposedWords.length === 0) {
    return { current: plainExcerpt(currentWords), proposed: plainExcerpt(proposedWords) };
  }

  let prefix = 0;
  while (
    prefix < currentWords.length &&
    prefix < proposedWords.length &&
    currentWords[prefix] === proposedWords[prefix]
  ) {
    prefix += 1;
  }
  let suffix = 0;
  while (
    suffix < currentWords.length - prefix &&
    suffix < proposedWords.length - prefix &&
    currentWords[currentWords.length - suffix - 1] === proposedWords[proposedWords.length - suffix - 1]
  ) {
    suffix += 1;
  }
  if (prefix === currentWords.length && prefix === proposedWords.length) {
    return { current: plainExcerpt(currentWords), proposed: plainExcerpt(proposedWords) };
  }
  return {
    current: excerpt(currentWords, prefix, currentWords.length - suffix),
    proposed: excerpt(proposedWords, prefix, proposedWords.length - suffix),
  };
}
