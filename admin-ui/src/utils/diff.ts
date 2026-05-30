// Word-level diff using Hunt-McIlroy LCS. Output is a list of segments
// labeled equal/added/removed so the UI can render a single visual diff
// without adding a runtime dependency for one workflow.

export type DiffOp = "equal" | "added" | "removed";

export interface DiffSegment {
  op: DiffOp;
  text: string;
}

export type DiffSide = "before" | "after";

interface Token {
  text: string;
  isWhitespace: boolean;
}

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  if (!input) return tokens;
  const regex = /\s+|\S+/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(input)) !== null) {
    const text = match[0];
    tokens.push({ text, isWhitespace: /^\s+$/.test(text) });
  }
  return tokens;
}

function buildLcsTable(a: Token[], b: Token[]): number[][] {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1].text === b[j - 1].text) {
        table[i][j] = table[i - 1][j - 1] + 1;
      } else {
        table[i][j] = Math.max(table[i - 1][j], table[i][j - 1]);
      }
    }
  }
  return table;
}

export function diffWords(before: string, after: string): DiffSegment[] {
  const a = tokenize(before);
  const b = tokenize(after);
  const table = buildLcsTable(a, b);

  const segments: DiffSegment[] = [];
  const push = (op: DiffOp, text: string) => {
    if (!text) return;
    const last = segments[segments.length - 1];
    if (last && last.op === op) {
      last.text += text;
    } else {
      segments.push({ op, text });
    }
  };

  let i = a.length;
  let j = b.length;
  const reverse: DiffSegment[] = [];
  while (i > 0 && j > 0) {
    if (a[i - 1].text === b[j - 1].text) {
      reverse.push({ op: "equal", text: a[i - 1].text });
      i -= 1;
      j -= 1;
    } else if (table[i - 1][j] >= table[i][j - 1]) {
      reverse.push({ op: "removed", text: a[i - 1].text });
      i -= 1;
    } else {
      reverse.push({ op: "added", text: b[j - 1].text });
      j -= 1;
    }
  }
  while (i > 0) {
    reverse.push({ op: "removed", text: a[i - 1].text });
    i -= 1;
  }
  while (j > 0) {
    reverse.push({ op: "added", text: b[j - 1].text });
    j -= 1;
  }

  for (let k = reverse.length - 1; k >= 0; k -= 1) {
    push(reverse[k].op, reverse[k].text);
  }
  return segments;
}

export function segmentsForSide(segments: DiffSegment[], side: DiffSide): DiffSegment[] {
  return segments.filter((segment) => (
    segment.op === "equal"
    || (side === "before" ? segment.op === "removed" : segment.op === "added")
  ));
}
