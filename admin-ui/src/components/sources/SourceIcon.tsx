import { cn } from "@/lib/utils";
import { AGENT_SESSION_CLIENT_MARK, BRAND_MARKS, SOURCE_DOT_FALLBACK, SOURCE_TYPE_MARKS } from "@/views/sources/sourceBrand";

const AGENT_SESSION_TYPE = "agent_session";

type SourceIconProps = {
  /** Source type, e.g. "confluence" or "agent_session". */
  type: string;
  /**
   * For agent_session sources, the plugin client identifier ("codex" or
   * "claude-code"). Picks exactly one brand mark. Omit for non-agent types.
   */
  client?: string | null;
  /** Sizing/positioning classes applied to each mark (e.g. "size-5"). */
  className?: string;
};

/**
 * Renders a source's brand logo. Agent-session sources show a single mark
 * chosen by `client` ("codex" or "claude-code"). All other source types use
 * the SOURCE_TYPE_MARKS table. Unknown types fall back to a colored dot so
 * future genes still render.
 */
export function SourceIcon({ type, client, className }: SourceIconProps) {
  // Agent session: resolve to a single brand mark based on the client, or fall
  // back to a neutral dot when the client is unrecognised or absent.
  if (type === AGENT_SESSION_TYPE) {
    const brandKey = client ? AGENT_SESSION_CLIENT_MARK[client] : undefined;
    if (!brandKey) {
      return (
        <span
          role="img"
          aria-label="Agent session"
          className={cn("inline-block size-2.5 rounded-full bg-muted-foreground", className)}
        />
      );
    }
    const mark = BRAND_MARKS[brandKey];
    return (
      <span role="img" aria-label={mark.label} title={mark.label} className="inline-flex items-center">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className={cn("shrink-0", mark.color ? undefined : "text-foreground", className)}
          style={mark.color ? { color: mark.color } : undefined}
          dangerouslySetInnerHTML={{ __html: mark.markup }}
        />
      </span>
    );
  }

  const keys = SOURCE_TYPE_MARKS[type];

  if (!keys) {
    return (
      <span
        role="img"
        aria-label={type}
        className={cn(
          "inline-block size-2.5 rounded-full",
          SOURCE_DOT_FALLBACK[type] ?? "bg-muted-foreground",
          className,
        )}
      />
    );
  }

  // The cluster announces once as a single labeled image; the inner marks are
  // decorative. `title` surfaces the same name as a hover tooltip.
  const label = keys.map((key) => BRAND_MARKS[key].label).join(", ");

  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={cn("inline-flex items-center", keys.length > 1 && "-space-x-1")}
    >
      {keys.map((key) => {
        const mark = BRAND_MARKS[key];
        // Markup is a static, bundled Simple Icons path (no user input), so
        // dangerouslySetInnerHTML carries no XSS risk and keeps the SVG verbatim.
        return (
          <svg
            key={key}
            aria-hidden="true"
            viewBox="0 0 24 24"
            className={cn("shrink-0", mark.color ? undefined : "text-foreground", className)}
            style={mark.color ? { color: mark.color } : undefined}
            dangerouslySetInnerHTML={{ __html: mark.markup }}
          />
        );
      })}
    </span>
  );
}
