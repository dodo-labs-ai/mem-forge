import { useRender } from "@base-ui/react/use-render";
import { ArrowLeft } from "lucide-react";
import type { ReactElement } from "react";

interface BackLinkProps {
  /** The page it returns to, such as "Memories". */
  label: string;
  /** The link element, such as the router's `<Link to="/memories" />`; the pattern fills in its content. */
  render: ReactElement;
}

/** The link above a detail page back to the list it belongs to. */
export function BackLink({ label, render }: BackLinkProps) {
  return useRender({
    defaultTagName: "a",
    render,
    props: {
      className: "inline-flex items-center gap-1.5 text-sm text-subtle-foreground hover:text-foreground",
      children: (
        <>
          <ArrowLeft aria-hidden className="size-3.5" />
          {label}
        </>
      ),
    },
  });
}
