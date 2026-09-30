import { Link } from "react-router-dom";
import { EmptyState } from "@/patterns";
import { Button } from "@/ui/button";

export function NotFoundPage() {
  return (
    <EmptyState
      title="This page does not exist"
      description="The link may be out of date."
      action={<Button nativeButton={false} render={<Link to="/" />}>Go to the start page</Button>}
    />
  );
}
