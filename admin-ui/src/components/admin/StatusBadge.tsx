import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const memoryTypeStyles: Record<string, string> = {
  fact: "border-blue-200 bg-blue-50 text-blue-700",
  decision: "border-violet-200 bg-violet-50 text-violet-700",
  convention: "border-emerald-200 bg-emerald-50 text-emerald-700",
  procedure: "border-orange-200 bg-orange-50 text-orange-700",
};

const statusStyles: Record<string, string> = {
  active: "bg-emerald-500",
  pending_review: "bg-amber-500",
  superseded: "bg-blue-500",
  retired: "bg-muted-foreground",
  decayed: "bg-muted-foreground",
};

export function MemoryTypeBadge({ type }: { type: string }) {
  return (
    <Badge variant="outline" className={cn("capitalize", memoryTypeStyles[type])}>
      {type}
    </Badge>
  );
}

export function StatusDot({ status }: { status: string }) {
  return (
    <span
      className={cn("inline-block size-2 rounded-full", statusStyles[status] ?? "bg-muted-foreground")}
      title={status}
    />
  );
}

export function ConfidenceBadge({ confidence }: { confidence: number }) {
  const label = confidence >= 0.8 ? "Strong" : confidence >= 0.55 ? "Moderate" : "Low";
  return <Badge variant="secondary">{label}</Badge>;
}
