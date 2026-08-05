export interface ReviewCueExcerpt {
  before: string;
  changed: string;
  after: string;
}

export interface ReviewChangeCue {
  current: ReviewCueExcerpt | null;
  proposed: ReviewCueExcerpt | null;
}

const CONTEXT_WORDS = 4;
const CHANGED_WORDS = 8;

function words(value: string | null | undefined): string[] {
  return value?.trim().split(/\s+/).filter(Boolean) ?? [];
}

function excerpt(tokens: string[], start: number, end: number): ReviewCueExcerpt | null {
  if (tokens.length === 0) return null;

  const changedEnd = Math.min(end, start + CHANGED_WORDS);
  const beforeStart = Math.max(0, start - CONTEXT_WORDS);
  const afterEnd = Math.min(tokens.length, changedEnd + CONTEXT_WORDS);
  return {
    before: `${beforeStart > 0 ? "… " : ""}${tokens.slice(beforeStart, start).join(" ")}`,
    changed: tokens.slice(start, changedEnd).join(" "),
    after: `${tokens.slice(changedEnd, afterEnd).join(" ")}${afterEnd < tokens.length ? " …" : ""}`,
  };
}

function plainExcerpt(tokens: string[]): ReviewCueExcerpt | null {
  if (tokens.length === 0) return null;
  const end = Math.min(tokens.length, CHANGED_WORDS + CONTEXT_WORDS);
  return {
    before: `${tokens.slice(0, end).join(" ")}${end < tokens.length ? " …" : ""}`,
    changed: "",
    after: "",
  };
}

export function buildReviewChangeCue(
  currentValue: string | null | undefined,
  proposedValue: string | null | undefined,
): ReviewChangeCue {
  const current = words(currentValue);
  const proposed = words(proposedValue);

  if (current.length === 0 || proposed.length === 0) {
    return {
      current: plainExcerpt(current),
      proposed: plainExcerpt(proposed),
    };
  }

  let prefix = 0;
  while (
    prefix < current.length &&
    prefix < proposed.length &&
    current[prefix] === proposed[prefix]
  ) {
    prefix += 1;
  }

  let suffix = 0;
  while (
    suffix < current.length - prefix &&
    suffix < proposed.length - prefix &&
    current[current.length - suffix - 1] === proposed[proposed.length - suffix - 1]
  ) {
    suffix += 1;
  }

  if (prefix === current.length && prefix === proposed.length) {
    return {
      current: plainExcerpt(current),
      proposed: plainExcerpt(proposed),
    };
  }

  return {
    current: excerpt(current, prefix, current.length - suffix),
    proposed: excerpt(proposed, prefix, proposed.length - suffix),
  };
}
