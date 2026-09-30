import { Link } from "react-router-dom";
import { memoryPath } from "@/lib/paths";
import { formatDateTime } from "@/lib/format";
import { PropertyList, type Property } from "@/patterns";
import { useMemory } from "./api";
import { DetailCard } from "./DetailCard";
import type { LifecycleDetail } from "./model/lifecycle";

/** The newer memory by its statement once it loads, by id until then. */
function ReplacementLink({ memoryId }: { memoryId: string }) {
  const replacement = useMemory(memoryId);
  return (
    <Link to={memoryPath(memoryId)} className="font-medium underline-offset-2 hover:underline">
      {replacement.data?.content ?? memoryId}
    </Link>
  );
}

export function LifecycleCard({ detail }: { detail: LifecycleDetail }) {
  const items: Property[] = [
    { label: "Status", value: detail.status },
    { label: "Reason", value: detail.reason },
    ...(detail.occurred ? [{ label: detail.occurred.label, value: formatDateTime(detail.occurred.at) }] : []),
    ...(detail.replacedBy ? [{ label: "Replaced by", value: <ReplacementLink memoryId={detail.replacedBy} /> }] : []),
    ...(detail.technicalReason
      ? [{ label: "Technical reason", value: <span className="font-mono text-xs text-muted-foreground">{detail.technicalReason}</span> }]
      : []),
  ];
  return (
    <DetailCard title="Lifecycle">
      <PropertyList items={items} />
    </DetailCard>
  );
}
