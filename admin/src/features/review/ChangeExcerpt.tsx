import type { ChangeCueExcerpt } from "./model/changeCue";

/** One side of a change with the words that differ highlighted. */
export function ChangeExcerpt({ excerpt, emptyText }: { excerpt: ChangeCueExcerpt | null; emptyText: string }) {
  if (excerpt === null) return <span className="text-muted-foreground">{emptyText}</span>;
  const { before, changed, after } = excerpt;
  return (
    <span className="text-subtle-foreground">
      {before}
      {before && changed ? " " : null}
      {changed ? <mark className="rounded-sm bg-tone-warn-soft px-0.5 text-foreground">{changed}</mark> : null}
      {changed && after ? " " : null}
      {after}
    </span>
  );
}
