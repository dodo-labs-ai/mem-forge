import { Archive, ArrowRight, Info, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { memoryPath } from "@/lib/paths";
import { Notice, type Tone } from "@/patterns";
import { buttonVariants } from "@/ui/button";
import type { StatusBanner } from "./model/lifecycle";

const BANNER_ICONS: Partial<Record<Tone, typeof Info>> = { idle: Archive, warn: ShieldCheck };

/** What a non-active status means for search, at the top of the memory. */
export function MemoryStatusBanner({ banner }: { banner: StatusBanner }) {
  return (
    <Notice
      tone={banner.tone}
      icon={BANNER_ICONS[banner.tone] ?? Info}
      title={banner.title}
      action={
        banner.newerMemoryId ? (
          <Link to={memoryPath(banner.newerMemoryId)} className={buttonVariants({ size: "sm", variant: "outline" })}>
            <ArrowRight />
            Open newer memory
          </Link>
        ) : null
      }
    >
      {banner.message}
    </Notice>
  );
}
